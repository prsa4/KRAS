import copy
import functools
import json
import math
import random
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from urllib import error, parse, request

from faker import Faker

from model_OOP import CargoGeneratorService


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


@dataclass
class SimulationSettings:
    sim_time: int
    workers_n: int
    delay: float

    @classmethod
    def from_input(cls, input_fn=input):
        return cls(
            sim_time=int(input_fn("Время: ")),
            workers_n=int(input_fn("Воркеры: ")),
            delay=float(input_fn("Задержка: ")),
        )


@dataclass
class SimulationConfig:
    base_dir: Path
    output_dir: Path
    map_file: Path
    map_data_file: Path
    map_template_file: Path
    map_url_path: str
    map_refresh_seconds: float = MAP_REFRESH_SECONDS
    default_map_center: tuple = DEFAULT_MAP_CENTER
    osrm_route_url: str = OSRM_ROUTE_URL
    osrm_min_request_interval: float = OSRM_MIN_REQUEST_INTERVAL
    osrm_retry_attempts: int = OSRM_RETRY_ATTEMPTS
    country_pool: list = field(default_factory=lambda: list(COUNTRY_POOL))
    transit_step_km: int = TRANSIT_STEP_KM

    @classmethod
    def from_script_path(cls, script_path):
        base_dir = Path(script_path).resolve().parent
        output_dir = base_dir / Path(script_path).stem
        map_file = output_dir / "shipments_map_09_OOP.html"
        return cls(
            base_dir=base_dir,
            output_dir=output_dir,
            map_file=map_file,
            map_data_file=output_dir / "shipments_data_09_OOP.json",
            map_template_file=base_dir / "shipments_map_template_09.html",
            map_url_path=map_file.relative_to(base_dir).as_posix(),
        )


@dataclass
class SimulationStats:
    processed: int = 0
    delivered: int = 0
    failed: int = 0
    events: int = 0
    total_cost: float = 0.0
    total_delay_hours: float = 0.0

    def copy(self):
        return {
            "processed": self.processed,
            "delivered": self.delivered,
            "failed": self.failed,
            "events": self.events,
            "total_cost": self.total_cost,
            "total_delay_hours": self.total_delay_hours,
        }


@dataclass
class SimulationRuntime:
    fake_service: object
    queue: Queue = field(default_factory=Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    map_stop_event: threading.Event = field(default_factory=threading.Event)
    stats: SimulationStats = field(default_factory=SimulationStats)
    packages: list = field(default_factory=list)
    workers: list = field(default_factory=list)
    stats_lock: threading.Lock = field(default_factory=threading.Lock)
    workers_lock: threading.Lock = field(default_factory=threading.Lock)
    packages_lock: threading.Lock = field(default_factory=threading.Lock)


class RouteService:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.next_request_at = 0.0

    @staticmethod
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

    def wait_for_route_slot(self):
        with self.request_lock:
            now = time.monotonic()
            wait_time = self.next_request_at - now

            if wait_time > 0:
                time.sleep(wait_time)

            self.next_request_at = (
                time.monotonic() + self.config.osrm_min_request_interval
            )

    def request_route_data(self, lat1, lon1, lat2, lon2):
        query = parse.urlencode(
            {
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "annotations": "false",
            }
        )
        coordinates = f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
        url = f"{self.config.osrm_route_url}/{coordinates}?{query}"

        for attempt in range(self.config.osrm_retry_attempts):
            self.wait_for_route_slot()
            response = self.get_json(url)

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
                        delay = self.config.osrm_min_request_interval * (attempt + 1)
                else:
                    delay = self.config.osrm_min_request_interval * (attempt + 1)

                warn(
                    f"OSRM повтор | попытка={attempt + 1} | пауза={delay:.1f} сек"
                )
                time.sleep(delay)
                continue

            break

        return None

    @staticmethod
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
            lat = ((1 - t) ** 2) * lat1 + 2 * (1 - t) * t * control_lat + (t ** 2) * lat2
            lon = ((1 - t) ** 2) * lon1 + 2 * (1 - t) * t * control_lon + (t ** 2) * lon2
            points.append([round(lat, 6), round(lon, 6)])

        return points

    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        radius = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)
        ) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c

    def split_route_into_steps(self, route_points, target_step_km):
        if len(route_points) < 2:
            return route_points

        total_distance = 0.0

        for index in range(1, len(route_points)):
            start_lat, start_lon = route_points[index - 1]
            end_lat, end_lon = route_points[index]
            total_distance += self.calculate_distance(start_lat, start_lon, end_lat, end_lon)

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
                segment_distance = self.calculate_distance(
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

    def get_route(self, lat1, lon1, lat2, lon2):
        cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))

        with self.cache_lock:
            cached_route = self.cache.get(cache_key)

        if cached_route is not None:
            return cached_route

        data = self.request_route_data(lat1, lon1, lat2, lon2)

        if not data or not data.get("routes"):
            warn(f"OSRM fallback | маршрут={lat1:.2f},{lon1:.2f} -> {lat2:.2f},{lon2:.2f}")
            route = {
                "points": self.build_fallback_route(lat1, lon1, lat2, lon2),
                "is_fallback": True,
            }
        else:
            coordinates = data["routes"][0]["geometry"]["coordinates"]
            route = {
                "points": [[lat, lon] for lon, lat in coordinates],
                "is_fallback": False,
            }

        with self.cache_lock:
            self.cache[cache_key] = route

        return route


class PackageFactory:
    def __init__(self, fake_service, rng, country_pool, timestamp_supplier, cargo_service):
        self.fake_service = fake_service
        self.rng = rng
        self.country_pool = country_pool
        self.timestamp_supplier = timestamp_supplier
        self.cargo_service = cargo_service

    def choose_destination(self):
        country = self.rng.choice(self.country_pool)
        latitude, longitude, city, _, _ = self.fake_service.local_latlng(
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

    def choose_route_points(self):
        origin, origin_coords = self.choose_destination()
        destination, destination_coords = self.choose_destination()

        while (
            origin["label"] == destination["label"]
            and origin["country"] == destination["country"]
        ):
            destination, destination_coords = self.choose_destination()

        return origin, origin_coords, destination, destination_coords

    def create_package(self):
        origin, (start_lat, start_lon), destination, (end_lat, end_lon) = (
            self.choose_route_points()
        )
        priority = self.rng.choice([0, 1])
        created_at = self.timestamp_supplier()
        cargo = self.cargo_service.generate_random_cargo(self.rng)

        return {
            "id": self.rng.randint(1000, 9999),
            "weight": cargo["estimated_weight_kg"],
            "origin": origin["label"],
            "origin_country": origin["country"],
            "destination": destination["label"],
            "country": destination["country"],
            "cargo": cargo,
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
                    "message": f"Посылка создана: {cargo['summary']}. Ожидает отправку.",
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


class FedExSimulator:
    def __init__(
        self,
        settings=None,
        script_path=__file__,
        fake_factory=Faker,
        cargo_service=None,
        browser_opener=webbrowser.open,
        rng=random,
    ):
        self.settings = settings or SimulationSettings.from_input()
        self.config = SimulationConfig.from_script_path(script_path)
        self.browser_opener = browser_opener
        self.rng = rng
        self.timestamp_supplier = current_time_text
        self.cargo_service = cargo_service or CargoGeneratorService()
        self.runtime = SimulationRuntime(fake_service=fake_factory())
        self.route_service = RouteService(self.config)
        self.package_factory = PackageFactory(
            self.runtime.fake_service,
            self.rng,
            self.config.country_pool,
            self.timestamp_supplier,
            self.cargo_service,
        )
        self.base_price = 10

    def calculate_cost(self, weight, priority, distance):
        multiplier = self.rng.uniform(1.5, 3)

        if priority == 1:
            multiplier *= 1.5

        return self.base_price + weight * multiplier + distance * 0.5

    def append_timeline_entry(self, package, message, entry_type="info"):
        package["timeline"].append(
            {
                "time": self.timestamp_supplier(),
                "type": entry_type,
                "message": message,
            }
        )

        if len(package["timeline"]) > MAX_TIMELINE_ENTRIES:
            package["timeline"] = package["timeline"][-MAX_TIMELINE_ENTRIES:]

    @staticmethod
    def estimate_delivery_hours(distance, priority, total_steps):
        speed_kmh = 82 if priority == 1 else 64
        handling_hours = 1.6 if priority == 1 else 3.2
        flow_hours = distance / speed_kmh
        checkpoint_hours = total_steps * (0.08 if priority == 1 else 0.12)
        return max(1.0, flow_hours + handling_hours + checkpoint_hours)

    @staticmethod
    def get_checkpoint_marks(total_steps):
        if total_steps <= 0:
            return set()

        return {
            max(1, round(total_steps * 0.25)),
            max(1, round(total_steps * 0.5)),
            max(1, round(total_steps * 0.75)),
            total_steps,
        }

    def choose_event_type(self, package, step_distance_km):
        if package["event_cooldown_steps"] > 0:
            return None

        risk = package["event_risk"]
        if risk <= 0.07:
            return None

        probability = min(0.28, 0.015 + risk * 0.58 + step_distance_km / 9000)
        if self.rng.random() > probability:
            return None

        available_types = [
            event_type
            for event_type in EVENT_TYPES
            if risk >= event_type["min_risk"]
        ]
        if not available_types:
            return None

        total_weight = sum(event_type["weight"] for event_type in available_types)
        pick = self.rng.uniform(0, total_weight)
        cursor = 0.0

        for event_type in available_types:
            cursor += event_type["weight"]
            if pick <= cursor:
                return event_type

        return available_types[-1]

    def build_event(self, event_type, package, step_index, step_distance_km):
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
            event["delay_hours"] = round(self.rng.uniform(1.5, 5.0), 1)
            event["cost_delta"] = round(self.rng.uniform(4, 12), 2)
            event["message"] = (
                f"{label}: плохая погода замедлила доставку на {event['delay_hours']:.1f} ч."
            )
        elif event_type["key"] == "rough_handling":
            event["delay_hours"] = round(self.rng.uniform(0.4, 1.6), 1)
            event["cost_delta"] = round(self.rng.uniform(3, 8), 2)
            event["integrity_delta"] = -self.rng.randint(4, 12)
            event["damage_delta"] = 1
            event["message"] = (
                f"{label}: целостность снизилась на {abs(event['integrity_delta'])}% ."
            )
        elif event_type["key"] == "moisture_damage":
            event["delay_hours"] = round(self.rng.uniform(0.5, 2.2), 1)
            event["cost_delta"] = round(self.rng.uniform(4, 10), 2)
            event["integrity_delta"] = -self.rng.randint(5, 14)
            event["damage_delta"] = 1
            event["message"] = (
                f"{label}: коробка отсырела, целостность снизилась на {abs(event['integrity_delta'])}% ."
            )
        elif event_type["key"] == "vehicle_breakdown":
            event["delay_hours"] = round(self.rng.uniform(2.0, 6.0), 1)
            event["cost_delta"] = round(self.rng.uniform(8, 20), 2)
            event["message"] = (
                f"{label}: потребовался ремонт, задержка {event['delay_hours']:.1f} ч."
            )
        elif event_type["key"] == "route_detour":
            event["delay_hours"] = round(self.rng.uniform(0.8, 2.5), 1)
            event["cost_delta"] = round(self.rng.uniform(6, 14), 2)
            event["message"] = (
                f"{label}: маршрут удлинен примерно на "
                f"{max(8, round(step_distance_km * self.rng.uniform(0.3, 0.7)))} км."
            )
        elif event_type["key"] == "cargo_shift":
            event["delay_hours"] = round(self.rng.uniform(0.6, 1.8), 1)
            event["cost_delta"] = round(self.rng.uniform(5, 11), 2)
            event["integrity_delta"] = -self.rng.randint(3, 9)
            event["damage_delta"] = 1
            event["message"] = (
                f"{label}: часть груза сместилась, целостность снизилась на {abs(event['integrity_delta'])}% ."
            )
        elif event_type["key"] == "attempted_theft":
            event["delay_hours"] = round(self.rng.uniform(1.0, 3.0), 1)
            event["cost_delta"] = round(self.rng.uniform(8, 18), 2)
            event["integrity_delta"] = -self.rng.randint(3, 10)
            event["damage_delta"] = 1 if self.rng.random() < 0.25 else 0
            if self.rng.random() < 0.08:
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

    def apply_event(self, package, event):
        package["events"].append(
            {
                "time": self.timestamp_supplier(),
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

        self.append_timeline_entry(package, event["message"], entry_type="event")

        with self.runtime.stats_lock:
            self.runtime.stats.events += 1

    def finalize_package_stats(self, package, succeeded):
        with self.runtime.stats_lock:
            self.runtime.stats.processed += 1
            self.runtime.stats.total_cost += package["cost"] or 0
            self.runtime.stats.total_delay_hours += package["delay_hours"]
            if succeeded:
                self.runtime.stats.delivered += 1
            else:
                self.runtime.stats.failed += 1

    def process_package(self, package):
        start_lat, start_lon = package["start_coords"]
        end_lat, end_lon = package["end_coords"]

        if start_lat is None or start_lon is None or end_lat is None or end_lon is None:
            package["status"] = "geocode_failed"
            package["interruption_reason"] = "Ошибка координат"
            self.append_timeline_entry(
                package,
                "Координаты недоступны, маршрут не построен.",
                entry_type="error",
            )
            self.finalize_package_stats(package, False)
            return

        distance = self.route_service.calculate_distance(start_lat, start_lon, end_lat, end_lon)
        route_data = self.route_service.get_route(start_lat, start_lon, end_lat, end_lon)
        transit_points = self.route_service.split_route_into_steps(
            route_data["points"],
            self.config.transit_step_km,
        )
        total_steps = max(1, len(transit_points) - 1)
        expected_hours = self.estimate_delivery_hours(distance, package["priority"], total_steps)
        base_cost = self.calculate_cost(package["weight"], package["priority"], distance)
        checkpoint_marks = self.get_checkpoint_marks(total_steps)

        package["distance_km"] = round(distance, 1)
        package["route"] = route_data["points"]
        package["route_is_fallback"] = route_data["is_fallback"]
        package["total_steps"] = total_steps
        package["expected_delivery_hours"] = round(expected_hours, 1)
        package["actual_delivery_hours"] = 0.0
        package["base_cost"] = round(base_cost, 2)
        package["cost"] = round(base_cost, 2)
        package["dispatched_at"] = self.timestamp_supplier()
        package["status"] = "in_transit"

        self.append_timeline_entry(
            package,
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
            step_distance = self.route_service.calculate_distance(
                previous_lat,
                previous_lon,
                next_lat,
                next_lon,
            )
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

            event_type = self.choose_event_type(package, step_distance)
            if event_type is not None:
                event = self.build_event(event_type, package, step_index, step_distance)
                self.apply_event(package, event)
                package["event_risk"] *= 0.18

                if package["integrity_pct"] <= 0:
                    package["status"] = "destroyed"
                    package["interruption_reason"] = "Целостность упала до 0%."
                    self.append_timeline_entry(
                        package,
                        "Посылка разрушена и больше не может быть доставлена.",
                        entry_type="fail",
                    )
                    package["processed_at"] = self.timestamp_supplier()
                    package["completion_delta_hours"] = round(
                        package["actual_delivery_hours"] - package["expected_delivery_hours"],
                        1,
                    )
                    self.finalize_package_stats(package, False)
                    return

                if event["terminal"]:
                    package["status"] = event["terminal_status"]
                    package["interruption_reason"] = event["message"]
                    package["processed_at"] = self.timestamp_supplier()
                    package["completion_delta_hours"] = round(
                        package["actual_delivery_hours"] - package["expected_delivery_hours"],
                        1,
                    )
                    self.finalize_package_stats(package, False)
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
                self.append_timeline_entry(
                    package,
                    f"Контрольная точка: пройдено {package['progress_pct']:.0f}% маршрута.",
                    entry_type="checkpoint",
                )

            time.sleep(0.18 if package["priority"] == 1 else 0.24)

        package["coords"] = package["end_coords"]
        package["progress_pct"] = 100.0
        package["status"] = "delivered"
        package["processed_at"] = self.timestamp_supplier()
        package["completion_delta_hours"] = round(
            package["actual_delivery_hours"] - package["expected_delivery_hours"],
            1,
        )
        self.append_timeline_entry(
            package,
            f"Доставка завершена. Факт: {package['actual_delivery_hours']:.1f} ч.",
            entry_type="done",
        )
        self.finalize_package_stats(package, True)

        info(
            f"Доставлена | id={package['id']} | факт={package['actual_delivery_hours']:.1f} ч | "
            f"отклонение={package['completion_delta_hours']:+.1f} ч | стоимость={package['cost']:.2f}"
        )

    def worker_loop(self, name):
        speed = self.rng.uniform(0.8, 1.5)
        queue = self.runtime.queue
        stop_event = self.runtime.stop_event

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
            self.process_package(package)

            time.sleep(0.5 / speed)
            queue.task_done()

        info(f"{name} остановлен")

    def adjust_delay(self, base_delay):
        size = self.runtime.queue.qsize()

        if size > 15:
            return base_delay * 1.4
        if size < 5:
            return max(0.2, base_delay * 0.75)

        return base_delay

    def generator_loop(self):
        current_delay = self.settings.delay
        queue = self.runtime.queue
        stop_event = self.runtime.stop_event

        while not stop_event.is_set():
            package = self.package_factory.create_package()

            with self.runtime.packages_lock:
                self.runtime.packages.append(package)

            info(
                f"Создана | id={package['id']} | {package['origin']} -> {package['destination']} | "
                f"тип={package['shipment_type']}"
            )

            queue.put(package)
            current_delay = self.adjust_delay(self.settings.delay)
            time.sleep(current_delay)

    def summarize_statuses(self, packages):
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

    def monitor_loop(self):
        queue = self.runtime.queue
        stop_event = self.runtime.stop_event

        while not stop_event.is_set():
            with self.runtime.stats_lock:
                processed = self.runtime.stats.processed
                delivered = self.runtime.stats.delivered
                failed = self.runtime.stats.failed
                total_cost = self.runtime.stats.total_cost
                events = self.runtime.stats.events

            info(
                f"Монитор | очередь={queue.qsize()} | завершено={processed} | доставлено={delivered} | "
                f"срывы={failed} | события={events} | сумма={total_cost:.2f}"
            )

            time.sleep(2)

    def start_worker(self, name):
        thread = threading.Thread(target=self.worker_loop, args=(name,))
        thread.start()
        return thread

    def manager_loop(self, max_workers):
        queue = self.runtime.queue
        stop_event = self.runtime.stop_event

        while not stop_event.is_set():
            if queue.qsize() > 10:
                with self.runtime.workers_lock:
                    if len(self.runtime.workers) < max_workers:
                        name = f"Extra-{len(self.runtime.workers)}"
                        thread = self.start_worker(name)
                        self.runtime.workers.append(thread)
                        info(f"Добавлен воркер | имя={name}")

            time.sleep(3)

    @staticmethod
    def serialize_package(package):
        return {
            "id": package["id"],
            "origin": package["origin"],
            "origin_country": package["origin_country"],
            "destination": package["destination"],
            "country": package["country"],
            "cargo": package["cargo"],
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

    def serialize_packages(self, packages):
        return list(map(self.serialize_package, packages))

    @staticmethod
    def iter_package_points(packages):
        for package in packages:
            yield package["start_coords"]
            yield package["coords"]
            yield package["end_coords"]

    def get_map_center(self, packages):
        points = [
            (lat, lon)
            for lat, lon in self.iter_package_points(packages)
            if lat is not None and lon is not None
        ]

        if not points:
            return {
                "lat": self.config.default_map_center[0],
                "lon": self.config.default_map_center[1],
            }

        lat_sum = sum(lat for lat, _ in points)
        lon_sum = sum(lon for _, lon in points)
        points_count = len(points)
        return {
            "lat": lat_sum / points_count,
            "lon": lon_sum / points_count,
        }

    def build_map_payload(self, packages, stats, simulation_active):
        status_summary = self.summarize_statuses(packages)
        return {
            "map_center": self.get_map_center(packages),
            "packages": self.serialize_packages(packages),
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

    def build_map_html(self, payload):
        template = self.config.map_template_file.read_text(encoding="utf-8")
        return template.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))

    def snapshot_state(self):
        with self.runtime.packages_lock:
            packages_copy = copy.deepcopy(self.runtime.packages)

        with self.runtime.stats_lock:
            stats_copy = self.runtime.stats.copy()

        return packages_copy, stats_copy

    def publish_map_state(self, packages, stats, simulation_active):
        payload = self.build_map_payload(packages, stats, simulation_active)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.map_file.write_text(
            self.build_map_html(payload),
            encoding="utf-8",
        )
        self.config.map_data_file.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def map_publisher_loop(self):
        while not self.runtime.map_stop_event.is_set():
            packages_copy, stats_copy = self.snapshot_state()
            self.publish_map_state(packages_copy, stats_copy, simulation_active=True)
            time.sleep(self.config.map_refresh_seconds)

    @staticmethod
    def make_quiet_handler(directory):
        class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                return

        return functools.partial(QuietHTTPRequestHandler, directory=str(directory))

    def start_map_server(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            self.make_quiet_handler(self.config.base_dir),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def start_initial_workers(self):
        started_workers = []

        for index in range(self.settings.workers_n):
            thread = self.start_worker(f"W{index}")
            started_workers.append(thread)

        with self.runtime.workers_lock:
            self.runtime.workers.extend(started_workers)

    def start_background_threads(self):
        cargo_thread = threading.Thread(
            target=self.cargo_service.cargo_prefetch_loop,
            args=(self.runtime.stop_event,),
            kwargs={"rng": self.rng},
            daemon=True,
        )
        generator_thread = threading.Thread(target=self.generator_loop)
        monitor_thread = threading.Thread(target=self.monitor_loop)
        map_thread = threading.Thread(target=self.map_publisher_loop, daemon=True)
        manager_thread = threading.Thread(
            target=self.manager_loop,
            args=(self.settings.workers_n + 3,),
        )

        cargo_thread.start()
        generator_thread.start()
        monitor_thread.start()
        map_thread.start()
        manager_thread.start()

        return cargo_thread, generator_thread, monitor_thread, map_thread, manager_thread

    def stop_workers(self):
        with self.runtime.workers_lock:
            active_workers = list(self.runtime.workers)

        for _ in active_workers:
            self.runtime.queue.put(None)

        for worker_thread in active_workers:
            worker_thread.join()

    def finalize_map(self, server, server_thread, map_thread):
        packages_copy, stats_copy = self.snapshot_state()
        self.publish_map_state(packages_copy, stats_copy, simulation_active=False)
        self.runtime.map_stop_event.set()
        map_thread.join(timeout=self.config.map_refresh_seconds + 1)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        info(f"Карта сохранена | файл={self.config.map_file}")

    def warm_up_cargo_buffer(self):
        info("Генерация товаров")
        self.cargo_service.prefill_cargo_buffer(target_size=1, rng=self.rng, max_rounds=1)
        info(f"Стартовый буфер товаров | элементов={self.cargo_service.get_cargo_buffer_size()}")

    def run(self):
        self.warm_up_cargo_buffer()

        info(
            "Старт симуляции v0.9 OOP | "
            f"время={self.settings.sim_time} сек | воркеры={self.settings.workers_n} | "
            f"задержка генерации={self.settings.delay:.2f} сек"
        )

        server, server_thread = self.start_map_server()
        map_url = f"http://127.0.0.1:{server.server_port}/{self.config.map_url_path}"

        packages_copy, stats_copy = self.snapshot_state()
        self.publish_map_state(packages_copy, stats_copy, simulation_active=True)
        self.browser_opener(map_url)
        info(f"Карта запущена | url={map_url}")

        self.start_initial_workers()
        cargo_thread, generator_thread, monitor_thread, map_thread, manager_thread = (
            self.start_background_threads()
        )

        time.sleep(self.settings.sim_time)
        self.runtime.stop_event.set()
        generator_thread.join()
        self.runtime.queue.join()
        self.stop_workers()
        monitor_thread.join()
        manager_thread.join()
        cargo_thread.join(timeout=2)
        self.finalize_map(server, server_thread, map_thread)
        info("Симуляция завершена")


def run(settings=None):
    simulator = FedExSimulator(settings=settings)
    simulator.run()


def main():
    run()


if __name__ == "__main__":
    main()