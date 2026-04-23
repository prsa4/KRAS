import copy
import functools
import json
import math
import random
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from urllib import error, parse, request

from faker import Faker

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

MAP_REFRESH_SECONDS = 1.0
DEFAULT_MAP_CENTER = (54.5260, 15.2551)
OSRM_MIN_REQUEST_INTERVAL = 0.5
OSRM_RETRY_ATTEMPTS = 3
TRANSIT_STEP_KM = 85
MAX_ROUTE_STEPS = 180
MAX_TIMELINE_ENTRIES = 30
EVENT_COOLDOWN_STEPS = 2

COUNTRY_POOL = [
    {"code": "GB", "name": "United Kingdom"},
    {"code": "FR", "name": "France"},
    {"code": "DE", "name": "Germany"},
    {"code": "ES", "name": "Spain"},
    {"code": "IT", "name": "Italy"},
    {"code": "NL", "name": "Netherlands"},
    {"code": "BE", "name": "Belgium"},
    {"code": "CH", "name": "Switzerland"},
    {"code": "AT", "name": "Austria"},
    {"code": "PL", "name": "Poland"},
    {"code": "CZ", "name": "Czech Republic"},
    {"code": "DK", "name": "Denmark"},
    {"code": "SE", "name": "Sweden"},
    {"code": "NO", "name": "Norway"},
    {"code": "PT", "name": "Portugal"},
    {"code": "IE", "name": "Ireland"},
    {"code": "RU", "name": "Russia"},
#   {"code": "US", "name": "United States"},
]

EVENT_TYPES = [
    {
        "key": "weather_delay",
        "label": "Погодная задержка",
        "min_risk": 0.12,
        "weight": 2.4,
        "kind": "delay",
    },
    {
        "key": "rough_handling",
        "label": "Неаккуратная перегрузка",
        "min_risk": 0.18,
        "weight": 3.1,
        "kind": "damage",
    },
    {
        "key": "moisture_damage",
        "label": "Намокание упаковки",
        "min_risk": 0.16,
        "weight": 2.1,
        "kind": "damage",
    },
    {
        "key": "vehicle_breakdown",
        "label": "Поломка транспорта",
        "min_risk": 0.22,
        "weight": 1.8,
        "kind": "delay",
    },
    {
        "key": "route_detour",
        "label": "Объезд маршрута",
        "min_risk": 0.20,
        "weight": 1.5,
        "kind": "reroute",
    },
    {
        "key": "cargo_shift",
        "label": "Смещение груза",
        "min_risk": 0.24,
        "weight": 1.6,
        "kind": "damage",
    },
    {
        "key": "attempted_theft",
        "label": "Попытка кражи",
        "min_risk": 0.34,
        "weight": 0.9,
        "kind": "critical",
    },
    {
        "key": "parcel_lost",
        "label": "Потеря посылки",
        "min_risk": 0.58,
        "weight": 0.45,
        "kind": "terminal",
    },
]


def info(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def warn(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def current_time_text():
    return time.strftime("%H:%M:%S")


def make_config(script_path):
    base_dir = Path(script_path).resolve().parent
    output_dir = base_dir / Path(script_path).stem
    return {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "map_file": output_dir / "shipments_map_08.html",
        "map_data_file": output_dir / "shipments_data_08.json",
        "map_template_file": base_dir / "shipments_map_template_08.html",
        "map_url_path": (output_dir / "shipments_map_08.html").relative_to(base_dir).as_posix(),
        "map_refresh_seconds": MAP_REFRESH_SECONDS,
        "default_map_center": DEFAULT_MAP_CENTER,
        "osrm_route_url": OSRM_ROUTE_URL,
        "osrm_min_request_interval": OSRM_MIN_REQUEST_INTERVAL,
        "osrm_retry_attempts": OSRM_RETRY_ATTEMPTS,
        "country_pool": COUNTRY_POOL,
        "transit_step_km": TRANSIT_STEP_KM,
    }


def create_stats():
    return {
        "processed": 0,
        "delivered": 0,
        "failed": 0,
        "events": 0,
        "total_cost": 0.0,
        "total_delay_hours": 0.0,
    }


def create_route_state():
    return {
        "cache": {},
        "cache_lock": threading.Lock(),
        "request_lock": threading.Lock(),
        "next_request_at": 0.0,
    }


def create_runtime(fake_factory=Faker):
    return {
        "queue": Queue(),
        "stop_event": threading.Event(),
        "map_stop_event": threading.Event(),
        "stats": create_stats(),
        "packages": [],
        "workers": [],
        "stats_lock": threading.Lock(),
        "workers_lock": threading.Lock(),
        "packages_lock": threading.Lock(),
        "route_state": create_route_state(),
        "fake": fake_factory(),
    }


def get_json(url, timeout=10):
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=timeout) as response:
            return {
                "status": getattr(response, "status", 200),
                "body": json.loads(response.read().decode("utf-8")),
                "retry_after": None,
            }
    except error.HTTPError as exc:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        return {
            "status": exc.code,
            "body": body,
            "retry_after": retry_after,
        }
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def make_route_fetcher(config, route_state):
    def wait_for_route_slot():
        with route_state["request_lock"]:
            now = time.monotonic()
            wait_time = route_state["next_request_at"] - now

            if wait_time > 0:
                time.sleep(wait_time)

            route_state["next_request_at"] = (
                time.monotonic() + config["osrm_min_request_interval"]
            )

    def request_route_data(lat1, lon1, lat2, lon2):
        query = parse.urlencode(
            {
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "annotations": "false",
            }
        )
        coordinates = f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
        url = f"{config['osrm_route_url']}/{coordinates}?{query}"

        for attempt in range(config["osrm_retry_attempts"]):
            wait_for_route_slot()
            response = get_json(url)

            if (
                response
                and response.get("status") == 200
                and response.get("body", {}).get("code") == "Ok"
            ):
                return response.get("body")

            if response and response.get("status") in (429, 500, 502, 503, 504):
                retry_after = response.get("retry_after")

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = config["osrm_min_request_interval"] * (attempt + 1)
                else:
                    delay = config["osrm_min_request_interval"] * (attempt + 1)

                warn(
                    f"OSRM повтор | попытка={attempt + 1} | пауза={delay:.1f} сек"
                )
                time.sleep(delay)
                continue

            break

        return None

    def get_route(lat1, lon1, lat2, lon2):
        cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))

        with route_state["cache_lock"]:
            cached_route = route_state["cache"].get(cache_key)

        if cached_route is not None:
            return cached_route

        data = request_route_data(lat1, lon1, lat2, lon2)

        if not data or not data.get("routes"):
            warn(
                f"OSRM fallback | маршрут={lat1:.2f},{lon1:.2f} -> {lat2:.2f},{lon2:.2f}"
            )
            route = {
                "points": build_fallback_route(lat1, lon1, lat2, lon2),
                "is_fallback": True,
            }
        else:
            coordinates = data["routes"][0]["geometry"]["coordinates"]
            route = {
                "points": [[lat, lon] for lon, lat in coordinates],
                "is_fallback": False,
            }

        with route_state["cache_lock"]:
            route_state["cache"][cache_key] = route

        return route

    return get_route


def choose_destination(fake_service, rng, country_pool):
    country = rng.choice(country_pool)
    latitude, longitude, city, _, _ = fake_service.local_latlng(
        country_code=country["code"],
        coords_only=False,
    )
    return (
        {
            "label": city,
            "country": country["name"],
        },
        (float(latitude), float(longitude)),
    )


def retry_until_distinct(generator):
    def generate_pair():
        a, a_coords = generator()
        b, b_coords = generator()

        while a["label"] == b["label"] and a["country"] == b["country"]:
            b, b_coords = generator()

        return a, a_coords, b, b_coords

    return generate_pair


def build_fallback_route(lat1, lon1, lat2, lon2):
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    length = math.hypot(delta_lat, delta_lon)

    if length == 0:
        return [[lat1, lon1], [lat2, lon2]]

    midpoint_lat = (lat1 + lat2) / 2
    midpoint_lon = (lon1 + lon2) / 2
    offset = min(4.0, max(0.6, length * 0.18))
    perp_lat = -delta_lon / length
    perp_lon = delta_lat / length
    control_lat = midpoint_lat + perp_lat * offset
    control_lon = midpoint_lon + perp_lon * offset

    points = []

    for step in range(19):
        t = step / 18
        lat = ((1 - t) ** 2) * lat1 + 2 * (1 - t) * t * control_lat + (t**2) * lat2
        lon = ((1 - t) ** 2) * lon1 + 2 * (1 - t) * t * control_lon + (t**2) * lon2
        points.append([round(lat, 6), round(lon, 6)])

    return points


def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def split_route_into_steps(route_points, target_step_km):
    if len(route_points) < 2:
        return route_points

    total_distance = 0.0

    for index in range(1, len(route_points)):
        start_lat, start_lon = route_points[index - 1]
        end_lat, end_lon = route_points[index]
        total_distance += calculate_distance(start_lat, start_lon, end_lat, end_lon)

    effective_step_km = max(target_step_km, total_distance / MAX_ROUTE_STEPS)
    stepped_points = [list(route_points[0])]
    distance_since_checkpoint = 0.0
    checkpoint_origin = list(route_points[0])

    for index in range(1, len(route_points)):
        segment_start = (
            checkpoint_origin if distance_since_checkpoint > 0 else list(route_points[index - 1])
        )
        segment_end = list(route_points[index])

        while True:
            segment_distance = calculate_distance(
                segment_start[0],
                segment_start[1],
                segment_end[0],
                segment_end[1],
            )

            if segment_distance == 0:
                checkpoint_origin = segment_end
                break

            if distance_since_checkpoint + segment_distance < effective_step_km:
                distance_since_checkpoint += segment_distance
                checkpoint_origin = segment_end
                break

            remaining_to_step = effective_step_km - distance_since_checkpoint
            ratio = remaining_to_step / segment_distance
            lat = segment_start[0] + (segment_end[0] - segment_start[0]) * ratio
            lon = segment_start[1] + (segment_end[1] - segment_start[1]) * ratio
            next_point = [round(lat, 6), round(lon, 6)]
            stepped_points.append(next_point)
            segment_start = next_point
            checkpoint_origin = next_point
            distance_since_checkpoint = 0.0

    final_point = [round(route_points[-1][0], 6), round(route_points[-1][1], 6)]
    if stepped_points[-1] != final_point:
        stepped_points.append(final_point)

    return stepped_points


def make_package_factory(fake_service, rng, country_pool, timestamp_supplier):
    choose_route_points = retry_until_distinct(
        lambda: choose_destination(fake_service, rng, country_pool)
    )

    def create_package():
        origin, (start_lat, start_lon), destination, (end_lat, end_lon) = (
            choose_route_points()
        )
        priority = rng.choice([0, 1])
        created_at = timestamp_supplier()

        return {
            "id": rng.randint(1000, 9999),
            "weight": round(rng.uniform(0.5, 10), 2),
            "origin": origin["label"],
            "origin_country": origin["country"],
            "destination": destination["label"],
            "country": destination["country"],
            "start_coords": (start_lat, start_lon),
            "end_coords": (end_lat, end_lon),
            "coords": (start_lat, start_lon),
            "priority": priority,
            "shipment_type": "Экспресс" if priority == 1 else "Стандарт",
            "status": "queued",
            "distance_km": None,
            "base_cost": None,
            "cost": None,
            "extra_cost": 0.0,
            "route": [],
            "route_is_fallback": False,
            "progress_pct": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "integrity_pct": 100,
            "damage_count": 0,
            "event_risk": 0.0,
            "event_cooldown_steps": 0,
            "events": [],
            "timeline": [
                {
                    "time": created_at,
                    "type": "created",
                    "message": "Посылка создана и ожидает отправку.",
                }
            ],
            "created_at": created_at,
            "dispatched_at": None,
            "processed_at": None,
            "expected_delivery_hours": None,
            "actual_delivery_hours": 0.0,
            "delay_hours": 0.0,
            "completion_delta_hours": None,
            "interruption_reason": None,
            "last_event": None,
        }

    return create_package


def make_cost_calculator(base_price, rng):
    def calculate(weight, priority, distance):
        multiplier = rng.uniform(1.5, 3)

        if priority == 1:
            multiplier *= 1.5

        return base_price + weight * multiplier + distance * 0.5

    return calculate


def append_timeline_entry(package, timestamp_supplier, message, entry_type="info"):
    package["timeline"].append(
        {
            "time": timestamp_supplier(),
            "type": entry_type,
            "message": message,
        }
    )

    if len(package["timeline"]) > MAX_TIMELINE_ENTRIES:
        package["timeline"] = package["timeline"][-MAX_TIMELINE_ENTRIES:]


def estimate_delivery_hours(distance, priority, total_steps):
    speed_kmh = 82 if priority == 1 else 64
    handling_hours = 1.6 if priority == 1 else 3.2
    flow_hours = distance / speed_kmh
    checkpoint_hours = total_steps * (0.08 if priority == 1 else 0.12)
    return max(1.0, flow_hours + handling_hours + checkpoint_hours)


def get_checkpoint_marks(total_steps):
    if total_steps <= 0:
        return set()

    return {
        max(1, round(total_steps * 0.25)),
        max(1, round(total_steps * 0.5)),
        max(1, round(total_steps * 0.75)),
        total_steps,
    }


def choose_event_type(package, rng, step_distance_km):
    if package["event_cooldown_steps"] > 0:
        return None

    risk = package["event_risk"]
    if risk <= 0.07:
        return None

    probability = min(0.28, 0.015 + risk * 0.58 + step_distance_km / 9000)
    if rng.random() > probability:
        return None

    available_types = [
        event_type
        for event_type in EVENT_TYPES
        if risk >= event_type["min_risk"]
    ]
    if not available_types:
        return None

    total_weight = sum(event_type["weight"] for event_type in available_types)
    pick = rng.uniform(0, total_weight)
    cursor = 0.0

    for event_type in available_types:
        cursor += event_type["weight"]
        if pick <= cursor:
            return event_type

    return available_types[-1]


def build_event(event_type, package, rng, step_index, step_distance_km):
    label = event_type["label"]
    event = {
        "key": event_type["key"],
        "label": label,
        "step": step_index,
        "progress_pct": round((step_index / max(1, package["total_steps"])) * 100, 1),
        "delay_hours": 0.0,
        "cost_delta": 0.0,
        "integrity_delta": 0,
        "damage_delta": 0,
        "terminal": False,
        "terminal_status": None,
        "message": "",
    }

    if event_type["key"] == "weather_delay":
        event["delay_hours"] = round(rng.uniform(1.5, 5.0), 1)
        event["cost_delta"] = round(rng.uniform(4, 12), 2)
        event["message"] = (
            f"{label}: плохая погода замедлила доставку на {event['delay_hours']:.1f} ч."
        )
    elif event_type["key"] == "rough_handling":
        event["delay_hours"] = round(rng.uniform(0.4, 1.6), 1)
        event["cost_delta"] = round(rng.uniform(3, 8), 2)
        event["integrity_delta"] = -rng.randint(4, 12)
        event["damage_delta"] = 1
        event["message"] = (
            f"{label}: целостность снизилась на {abs(event['integrity_delta'])}% ."
        )
    elif event_type["key"] == "moisture_damage":
        event["delay_hours"] = round(rng.uniform(0.5, 2.2), 1)
        event["cost_delta"] = round(rng.uniform(4, 10), 2)
        event["integrity_delta"] = -rng.randint(5, 14)
        event["damage_delta"] = 1
        event["message"] = (
            f"{label}: коробка отсырела, целостность снизилась на {abs(event['integrity_delta'])}% ."
        )
    elif event_type["key"] == "vehicle_breakdown":
        event["delay_hours"] = round(rng.uniform(2.0, 6.0), 1)
        event["cost_delta"] = round(rng.uniform(8, 20), 2)
        event["message"] = (
            f"{label}: потребовался ремонт, задержка {event['delay_hours']:.1f} ч."
        )
    elif event_type["key"] == "route_detour":
        event["delay_hours"] = round(rng.uniform(0.8, 2.5), 1)
        event["cost_delta"] = round(rng.uniform(6, 14), 2)
        event["message"] = (
            f"{label}: маршрут удлинен примерно на {max(8, round(step_distance_km * rng.uniform(0.3, 0.7)))} км."
        )
    elif event_type["key"] == "cargo_shift":
        event["delay_hours"] = round(rng.uniform(0.6, 1.8), 1)
        event["cost_delta"] = round(rng.uniform(5, 11), 2)
        event["integrity_delta"] = -rng.randint(3, 9)
        event["damage_delta"] = 1
        event["message"] = (
            f"{label}: часть груза сместилась, целостность снизилась на {abs(event['integrity_delta'])}% ."
        )
    elif event_type["key"] == "attempted_theft":
        event["delay_hours"] = round(rng.uniform(1.0, 3.0), 1)
        event["cost_delta"] = round(rng.uniform(8, 18), 2)
        event["integrity_delta"] = -rng.randint(3, 10)
        event["damage_delta"] = 1 if rng.random() < 0.25 else 0
        if rng.random() < 0.08:
            event["terminal"] = True
            event["terminal_status"] = "stolen"
            event["message"] = "Кража: отправление исчезло во время транзита."
        else:
            event["message"] = "Попытка кражи сорвала график и повредила упаковку."
    elif event_type["key"] == "parcel_lost":
        event["terminal"] = True
        event["terminal_status"] = "lost"
        event["message"] = "Потеря посылки: груз не найден на следующем транзитном узле."

    return event


def apply_event(package, event, stats, stats_lock, timestamp_supplier):
    package["events"].append(
        {
            "time": timestamp_supplier(),
            **event,
        }
    )
    package["last_event"] = event["label"]
    package["delay_hours"] = round(package["delay_hours"] + event["delay_hours"], 1)
    package["actual_delivery_hours"] = round(
        package["actual_delivery_hours"] + event["delay_hours"],
        2,
    )
    package["extra_cost"] = round(package["extra_cost"] + event["cost_delta"], 2)
    package["cost"] = round((package["base_cost"] or 0) + package["extra_cost"], 2)
    package["integrity_pct"] = max(0, package["integrity_pct"] + event["integrity_delta"])
    package["damage_count"] += event["damage_delta"]
    package["event_cooldown_steps"] = EVENT_COOLDOWN_STEPS

    append_timeline_entry(package, timestamp_supplier, event["message"], entry_type="event")

    with stats_lock:
        stats["events"] += 1


def finalize_package_stats(package, stats, stats_lock, succeeded):
    with stats_lock:
        stats["processed"] += 1
        stats["total_cost"] += package["cost"] or 0
        stats["total_delay_hours"] += package["delay_hours"]
        if succeeded:
            stats["delivered"] += 1
        else:
            stats["failed"] += 1


def process_package(
    package,
    runtime,
    cost_function,
    route_fetcher,
    rng,
    timestamp_supplier,
    config,
):
    start_lat, start_lon = package["start_coords"]
    end_lat, end_lon = package["end_coords"]

    if start_lat is None or start_lon is None or end_lat is None or end_lon is None:
        package["status"] = "geocode_failed"
        package["interruption_reason"] = "Ошибка координат"
        append_timeline_entry(
            package,
            timestamp_supplier,
            "Координаты недоступны, маршрут не построен.",
            entry_type="error",
        )
        finalize_package_stats(package, runtime["stats"], runtime["stats_lock"], False)
        return

    distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    route_data = route_fetcher(start_lat, start_lon, end_lat, end_lon)
    transit_points = split_route_into_steps(route_data["points"], config["transit_step_km"])
    total_steps = max(1, len(transit_points) - 1)
    expected_hours = estimate_delivery_hours(distance, package["priority"], total_steps)
    base_cost = cost_function(package["weight"], package["priority"], distance)
    checkpoint_marks = get_checkpoint_marks(total_steps)

    package["distance_km"] = round(distance, 1)
    package["route"] = route_data["points"]
    package["route_is_fallback"] = route_data["is_fallback"]
    package["total_steps"] = total_steps
    package["expected_delivery_hours"] = round(expected_hours, 1)
    package["actual_delivery_hours"] = 0.0
    package["base_cost"] = round(base_cost, 2)
    package["cost"] = round(base_cost, 2)
    package["dispatched_at"] = timestamp_supplier()
    package["status"] = "in_transit"

    append_timeline_entry(
        package,
        timestamp_supplier,
        f"Посылка отправлена. План: {package['expected_delivery_hours']:.1f} ч, {total_steps} шагов.",
        entry_type="dispatch",
    )

    info(
        f"Отправлена | id={package['id']} | {package['origin']} -> {package['destination']} | "
        f"план={package['expected_delivery_hours']:.1f} ч | шагов={total_steps}"
    )

    for step_index in range(1, len(transit_points)):
        previous_lat, previous_lon = transit_points[step_index - 1]
        next_lat, next_lon = transit_points[step_index]
        step_distance = calculate_distance(previous_lat, previous_lon, next_lat, next_lon)
        base_speed = 82 if package["priority"] == 1 else 64
        base_step_hours = max(0.15, step_distance / base_speed)

        package["event_risk"] = min(
            0.42,
            package["event_risk"]
            + 0.007
            + step_distance / 5000
            + package["damage_count"] * 0.01
            + (100 - package["integrity_pct"]) / 5000,
        )

        event_type = choose_event_type(package, rng, step_distance)
        if event_type is not None:
            event = build_event(event_type, package, rng, step_index, step_distance)
            apply_event(package, event, runtime["stats"], runtime["stats_lock"], timestamp_supplier)
            package["event_risk"] *= 0.18

            if package["integrity_pct"] <= 0:
                package["status"] = "destroyed"
                package["interruption_reason"] = "Целостность упала до 0%."
                append_timeline_entry(
                    package,
                    timestamp_supplier,
                    "Посылка разрушена и больше не может быть доставлена.",
                    entry_type="fail",
                )
                package["processed_at"] = timestamp_supplier()
                package["completion_delta_hours"] = round(
                    package["actual_delivery_hours"] - package["expected_delivery_hours"],
                    1,
                )
                finalize_package_stats(
                    package,
                    runtime["stats"],
                    runtime["stats_lock"],
                    False,
                )
                return

            if event["terminal"]:
                package["status"] = event["terminal_status"]
                package["interruption_reason"] = event["message"]
                package["processed_at"] = timestamp_supplier()
                package["completion_delta_hours"] = round(
                    package["actual_delivery_hours"] - package["expected_delivery_hours"],
                    1,
                )
                finalize_package_stats(
                    package,
                    runtime["stats"],
                    runtime["stats_lock"],
                    False,
                )
                info(
                    f"Срыв доставки | id={package['id']} | статус={package['status']} | "
                    f"причина={package['interruption_reason']}"
                )
                return

            package["status"] = "delayed" if event["delay_hours"] > 0 else "in_transit"
        else:
            package["last_event"] = None
            if package["status"] == "delayed":
                package["status"] = "in_transit"

        if package["event_cooldown_steps"] > 0:
            package["event_cooldown_steps"] -= 1

        package["actual_delivery_hours"] = round(
            package["actual_delivery_hours"] + base_step_hours,
            2,
        )
        package["coords"] = (next_lat, next_lon)
        package["current_step"] = step_index
        package["progress_pct"] = round((step_index / total_steps) * 100, 1)
        package["cost"] = round((package["base_cost"] or 0) + package["extra_cost"], 2)

        if step_index in checkpoint_marks and step_index < total_steps:
            append_timeline_entry(
                package,
                timestamp_supplier,
                f"Контрольная точка: пройдено {package['progress_pct']:.0f}% маршрута.",
                entry_type="checkpoint",
            )

        time.sleep(0.18 if package["priority"] == 1 else 0.24)

    package["coords"] = package["end_coords"]
    package["progress_pct"] = 100.0
    package["status"] = "delivered"
    package["processed_at"] = timestamp_supplier()
    package["completion_delta_hours"] = round(
        package["actual_delivery_hours"] - package["expected_delivery_hours"],
        1,
    )
    append_timeline_entry(
        package,
        timestamp_supplier,
        f"Доставка завершена. Факт: {package['actual_delivery_hours']:.1f} ч.",
        entry_type="done",
    )
    finalize_package_stats(package, runtime["stats"], runtime["stats_lock"], True)

    info(
        f"Доставлена | id={package['id']} | факт={package['actual_delivery_hours']:.1f} ч | "
        f"отклонение={package['completion_delta_hours']:+.1f} ч | стоимость={package['cost']:.2f}"
    )


def worker_loop(
    name,
    runtime,
    cost_function,
    route_fetcher,
    rng,
    timestamp_supplier,
    config,
):
    speed = rng.uniform(0.8, 1.5)
    queue = runtime["queue"]
    stop_event = runtime["stop_event"]

    while True:
        try:
            package = queue.get(timeout=1)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if package is None:
            queue.task_done()
            break

        info(f"{name} взял посылку | id={package['id']}")
        process_package(
            package,
            runtime,
            cost_function,
            route_fetcher,
            rng,
            timestamp_supplier,
            config,
        )

        time.sleep(0.5 / speed)
        queue.task_done()

    info(f"{name} остановлен")


def adjust_delay(queue, base_delay):
    size = queue.qsize()

    if size > 15:
        return base_delay * 1.4
    if size < 5:
        return max(0.2, base_delay * 0.75)

    return base_delay


def generator_loop(runtime, base_delay, create_package):
    current_delay = base_delay
    queue = runtime["queue"]
    stop_event = runtime["stop_event"]

    while not stop_event.is_set():
        package = create_package()

        with runtime["packages_lock"]:
            runtime["packages"].append(package)

        info(
            f"Создана | id={package['id']} | {package['origin']} -> {package['destination']} | "
            f"тип={package['shipment_type']}"
        )

        queue.put(package)
        current_delay = adjust_delay(queue, base_delay)
        time.sleep(current_delay)


def summarize_statuses(packages):
    summary = {
        "queued": 0,
        "in_transit": 0,
        "delayed": 0,
        "delivered": 0,
        "failed": 0,
    }
    failed_statuses = {"lost", "stolen", "destroyed", "geocode_failed"}

    for package in packages:
        status = package["status"]
        if status in failed_statuses:
            summary["failed"] += 1
        elif status in summary:
            summary[status] += 1
        else:
            summary["in_transit"] += 1

    return summary


def monitor_loop(runtime):
    queue = runtime["queue"]
    stop_event = runtime["stop_event"]

    while not stop_event.is_set():
        with runtime["stats_lock"]:
            processed = runtime["stats"]["processed"]
            delivered = runtime["stats"]["delivered"]
            failed = runtime["stats"]["failed"]
            total_cost = runtime["stats"]["total_cost"]
            events = runtime["stats"]["events"]

        info(
            f"Монитор | очередь={queue.qsize()} | завершено={processed} | доставлено={delivered} | "
            f"срывы={failed} | события={events} | сумма={total_cost:.2f}"
        )

        time.sleep(2)


def start_worker(
    name,
    runtime,
    cost_function,
    route_fetcher,
    rng,
    timestamp_supplier,
    config,
):
    thread = threading.Thread(
        target=worker_loop,
        args=(
            name,
            runtime,
            cost_function,
            route_fetcher,
            rng,
            timestamp_supplier,
            config,
        ),
    )
    thread.start()
    return thread


def manager_loop(
    runtime,
    cost_function,
    route_fetcher,
    max_workers,
    rng,
    timestamp_supplier,
    config,
):
    queue = runtime["queue"]
    stop_event = runtime["stop_event"]

    while not stop_event.is_set():
        if queue.qsize() > 10:
            with runtime["workers_lock"]:
                if len(runtime["workers"]) < max_workers:
                    name = f"Extra-{len(runtime['workers'])}"
                    thread = start_worker(
                        name,
                        runtime,
                        cost_function,
                        route_fetcher,
                        rng,
                        timestamp_supplier,
                        config,
                    )
                    runtime["workers"].append(thread)
                    info(f"Добавлен воркер | имя={name}")

        time.sleep(3)


def serialize_package(package):
    return {
        "id": package["id"],
        "origin": package["origin"],
        "origin_country": package["origin_country"],
        "destination": package["destination"],
        "country": package["country"],
        "start_lat": package["start_coords"][0],
        "start_lon": package["start_coords"][1],
        "end_lat": package["end_coords"][0],
        "end_lon": package["end_coords"][1],
        "lat": package["coords"][0],
        "lon": package["coords"][1],
        "priority": package["priority"],
        "shipment_type": package["shipment_type"],
        "status": package["status"],
        "weight": package["weight"],
        "distance_km": package["distance_km"],
        "base_cost": package["base_cost"],
        "cost": package["cost"],
        "extra_cost": package["extra_cost"],
        "created_at": package["created_at"],
        "dispatched_at": package["dispatched_at"],
        "processed_at": package["processed_at"],
        "route": package["route"],
        "route_is_fallback": package["route_is_fallback"],
        "progress_pct": package["progress_pct"],
        "current_step": package["current_step"],
        "total_steps": package["total_steps"],
        "integrity_pct": package["integrity_pct"],
        "damage_count": package["damage_count"],
        "event_risk": round(package["event_risk"], 3),
        "event_cooldown_steps": package["event_cooldown_steps"],
        "events": package["events"],
        "timeline": package["timeline"],
        "expected_delivery_hours": package["expected_delivery_hours"],
        "actual_delivery_hours": package["actual_delivery_hours"],
        "delay_hours": package["delay_hours"],
        "completion_delta_hours": package["completion_delta_hours"],
        "interruption_reason": package["interruption_reason"],
        "last_event": package["last_event"],
    }


def serialize_packages(packages):
    return list(map(serialize_package, packages))


def iter_package_points(packages):
    for package in packages:
        yield package["start_coords"]
        yield package["coords"]
        yield package["end_coords"]


def get_map_center(packages, default_center):
    points = [
        (lat, lon)
        for lat, lon in iter_package_points(packages)
        if lat is not None and lon is not None
    ]

    if not points:
        return {"lat": default_center[0], "lon": default_center[1]}

    lat_sum = sum(lat for lat, _ in points)
    lon_sum = sum(lon for _, lon in points)
    points_count = len(points)
    return {
        "lat": lat_sum / points_count,
        "lon": lon_sum / points_count,
    }


def build_map_payload(packages, stats, simulation_active, default_center):
    status_summary = summarize_statuses(packages)
    return {
        "map_center": get_map_center(packages, default_center),
        "packages": serialize_packages(packages),
        "simulation_active": simulation_active,
        "stats": {
            "processed": stats["processed"],
            "delivered": stats["delivered"],
            "failed": stats["failed"],
            "events": stats["events"],
            "total_cost": round(stats["total_cost"], 2),
            "total_delay_hours": round(stats["total_delay_hours"], 1),
            "total": len(packages),
            "queued": status_summary["queued"],
            "in_transit": status_summary["in_transit"] + status_summary["delayed"],
            "delayed": status_summary["delayed"],
        },
    }


def build_map_html(template_file, payload):
    template = template_file.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))


def snapshot_state(runtime):
    with runtime["packages_lock"]:
        packages_copy = copy.deepcopy(runtime["packages"])

    with runtime["stats_lock"]:
        stats_copy = dict(runtime["stats"])

    return packages_copy, stats_copy


def publish_map_state(config, packages, stats, simulation_active):
    payload = build_map_payload(
        packages,
        stats,
        simulation_active,
        config["default_map_center"],
    )
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["map_file"].write_text(
        build_map_html(config["map_template_file"], payload),
        encoding="utf-8",
    )
    config["map_data_file"].write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def map_publisher_loop(runtime, config):
    while not runtime["map_stop_event"].is_set():
        packages_copy, stats_copy = snapshot_state(runtime)
        publish_map_state(config, packages_copy, stats_copy, simulation_active=True)
        time.sleep(config["map_refresh_seconds"])


def make_quiet_handler(directory):
    class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

    return functools.partial(QuietHTTPRequestHandler, directory=str(directory))


def start_map_server(directory):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_quiet_handler(directory))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def read_simulation_settings(input_fn=input):
    return {
        "sim_time": int(input_fn("Время: ")),
        "workers_n": int(input_fn("Воркеры: ")),
        "delay": float(input_fn("Задержка: ")),
    }


def start_initial_workers(
    runtime,
    workers_count,
    cost_function,
    route_fetcher,
    rng,
    timestamp_supplier,
    config,
):
    started_workers = []

    for index in range(workers_count):
        thread = start_worker(
            f"W{index}",
            runtime,
            cost_function,
            route_fetcher,
            rng,
            timestamp_supplier,
            config,
        )
        started_workers.append(thread)

    with runtime["workers_lock"]:
        runtime["workers"].extend(started_workers)


def start_background_threads(
    runtime,
    config,
    delay,
    create_package,
    cost_function,
    route_fetcher,
    workers_count,
    rng,
    timestamp_supplier,
):
    generator_thread = threading.Thread(
        target=generator_loop,
        args=(runtime, delay, create_package),
    )
    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(runtime,),
    )
    map_thread = threading.Thread(
        target=map_publisher_loop,
        args=(runtime, config),
        daemon=True,
    )
    manager_thread = threading.Thread(
        target=manager_loop,
        args=(
            runtime,
            cost_function,
            route_fetcher,
            workers_count + 3,
            rng,
            timestamp_supplier,
            config,
        ),
    )

    generator_thread.start()
    monitor_thread.start()
    map_thread.start()
    manager_thread.start()

    return generator_thread, monitor_thread, map_thread, manager_thread


def stop_workers(runtime):
    with runtime["workers_lock"]:
        active_workers = list(runtime["workers"])

    for _ in active_workers:
        runtime["queue"].put(None)

    for worker_thread in active_workers:
        worker_thread.join()


def finalize_map(runtime, config, server, server_thread, map_thread):
    packages_copy, stats_copy = snapshot_state(runtime)
    publish_map_state(config, packages_copy, stats_copy, simulation_active=False)
    runtime["map_stop_event"].set()
    map_thread.join(timeout=config["map_refresh_seconds"] + 1)
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=2)
    info(f"Карта сохранена | файл={config['map_file']}")


def run(settings=None):
    settings = settings or read_simulation_settings()
    config = make_config(__file__)
    runtime = create_runtime()
    rng = random
    timestamp_supplier = current_time_text

    info(
        "Старт симуляции 0.8 | "
        f"время={settings['sim_time']} сек | воркеры={settings['workers_n']} | "
        f"задержка генерации={settings['delay']:.2f} сек"
    )

    cost_function = make_cost_calculator(10, rng)
    create_package = make_package_factory(
        runtime["fake"],
        rng,
        config["country_pool"],
        timestamp_supplier,
    )
    route_fetcher = make_route_fetcher(config, runtime["route_state"])

    server, server_thread = start_map_server(config["base_dir"])
    map_url = f"http://127.0.0.1:{server.server_port}/{config['map_url_path']}"

    packages_copy, stats_copy = snapshot_state(runtime)
    publish_map_state(config, packages_copy, stats_copy, simulation_active=True)
    webbrowser.open(map_url)
    info(f"Карта запущена | url={map_url}")

    start_initial_workers(
        runtime,
        settings["workers_n"],
        cost_function,
        route_fetcher,
        rng,
        timestamp_supplier,
        config,
    )

    generator_thread, monitor_thread, map_thread, manager_thread = start_background_threads(
        runtime,
        config,
        settings["delay"],
        create_package,
        cost_function,
        route_fetcher,
        settings["workers_n"],
        rng,
        timestamp_supplier,
    )

    time.sleep(settings["sim_time"])
    runtime["stop_event"].set()
    generator_thread.join()
    runtime["queue"].join()
    stop_workers(runtime)
    monitor_thread.join()
    manager_thread.join()
    finalize_map(runtime, config, server, server_thread, map_thread)
    info("Симуляция завершена")


def main():
    run()


if __name__ == "__main__":
    main()