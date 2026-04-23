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

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / Path(__file__).stem
MAP_FILE = OUTPUT_DIR / "shipments_map.html"
MAP_DATA_FILE = OUTPUT_DIR / "shipments_data.json"
MAP_TEMPLATE_FILE = Path(__file__).with_name("shipments_map_template.html")
MAP_URL_PATH = MAP_FILE.relative_to(BASE_DIR).as_posix()
MAP_REFRESH_SECONDS = 1.5
DEFAULT_MAP_CENTER = (54.5260, 15.2551)
OSRM_MIN_REQUEST_INTERVAL = 0.5
OSRM_RETRY_ATTEMPTS = 3

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

route_cache = {}
fake = Faker()
route_request_lock = threading.Lock()
next_route_request_at = 0.0


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


def info(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def warn(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


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


def wait_for_route_slot():
    global next_route_request_at

    with route_request_lock:
        now = time.monotonic()
        wait_time = next_route_request_at - now

        if wait_time > 0:
            time.sleep(wait_time)

        next_route_request_at = time.monotonic() + OSRM_MIN_REQUEST_INTERVAL


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
    url = f"{OSRM_ROUTE_URL}/{coordinates}?{query}"

    for attempt in range(OSRM_RETRY_ATTEMPTS):
        wait_for_route_slot()
        response = get_json(url)

        if response and response.get("status") == 200 and response.get("body", {}).get("code") == "Ok":
            return response.get("body")

        if response and response.get("status") in (429, 500, 502, 503, 504):
            retry_after = response.get("retry_after")

            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = OSRM_MIN_REQUEST_INTERVAL * (attempt + 1)
            else:
                delay = OSRM_MIN_REQUEST_INTERVAL * (attempt + 1)

            warn(f"OSRM повтор | попытка={attempt + 1} | пауза={delay:.1f} сек")
            time.sleep(delay)
            continue

        break

    return None


def choose_destination():
    country = random.choice(COUNTRY_POOL)
    latitude, longitude, city, _, _ = fake.local_latlng(
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


def choose_route_points():
    start, start_coords = choose_destination()
    finish, finish_coords = choose_destination()

    while start["label"] == finish["label"] and start["country"] == finish["country"]:
        finish, finish_coords = choose_destination()

    return start, start_coords, finish, finish_coords


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


def get_route(lat1, lon1, lat2, lon2):
    cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))

    if cache_key in route_cache:
        return route_cache[cache_key]

    data = request_route_data(lat1, lon1, lat2, lon2)

    if not data or not data.get("routes"):
        warn(
            f"OSRM fallback | маршрут={lat1:.2f},{lon1:.2f} -> {lat2:.2f},{lon2:.2f}"
        )
        route = {
            "points": build_fallback_route(lat1, lon1, lat2, lon2),
            "is_fallback": True,
        }
        route_cache[cache_key] = route
        return route

    coordinates = data["routes"][0]["geometry"]["coordinates"]
    route = {
        "points": [[lat, lon] for lon, lat in coordinates],
        "is_fallback": False,
    }
    route_cache[cache_key] = route
    return route


def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def create_package():
    origin, (start_lat, start_lon), destination, (end_lat, end_lon) = choose_route_points()
    priority = random.choice([0, 1])

    return {
        "id": random.randint(1000, 9999),
        "weight": round(random.uniform(0.5, 10), 2),
        "origin": origin["label"],
        "origin_country": origin["country"],
        "destination": destination["label"],
        "country": destination["country"],
        "start_coords": (start_lat, start_lon),
        "coords": (end_lat, end_lon),
        "priority": priority,
        "shipment_type": "Экспресс" if priority == 1 else "Стандарт",
        "status": "queued",
        "distance_km": None,
        "cost": None,
        "route": [],
        "route_is_fallback": False,
        "created_at": time.strftime("%H:%M:%S"),
        "processed_at": None,
    }


def create_cost_calculator(base_price):
    def calculate(weight, priority, distance):
        multiplier = random.uniform(1.5, 3)

        if priority == 1:
            multiplier *= 1.5

        return base_price + weight * multiplier + distance * 0.5

    return calculate


def process_package(package, cost_function, stats, stats_lock):
    start_lat, start_lon = package["start_coords"]
    lat, lon = package["coords"]

    if start_lat is None or start_lon is None or lat is None or lon is None:
        package["status"] = "geocode_failed"
        return

    distance = calculate_distance(start_lat, start_lon, lat, lon)
    route_data = get_route(start_lat, start_lon, lat, lon)
    cost = cost_function(package["weight"], package["priority"], distance)

    package["distance_km"] = round(distance, 1)
    package["cost"] = round(cost, 2)
    package["route"] = route_data["points"]
    package["route_is_fallback"] = route_data["is_fallback"]
    package["status"] = "processed"
    package["processed_at"] = time.strftime("%H:%M:%S")

    info(
        f"Обработана | id={package['id']} | {package['origin']} -> {package['destination']} | "
        f"тип={package['shipment_type']} | расстояние={distance:.1f} км | стоимость={cost:.2f}"
    )

    time.sleep(random.uniform(0.3, 1.0))

    with stats_lock:
        stats["processed"] += 1
        stats["total_cost"] += cost


def worker(name, queue, cost_function, stats, stats_lock, stop_event):
    speed = random.uniform(0.8, 1.5)

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
        process_package(package, cost_function, stats, stats_lock)

        time.sleep(1 / speed)
        queue.task_done()

    info(f"{name} остановлен")


def adjust_delay(queue, base):
    size = queue.qsize()

    if size > 15:
        return base * 1.5
    if size < 5:
        return max(0.2, base * 0.7)

    return base


def generate(queue, stop_event, delay, packages, packages_lock):
    current = delay

    while not stop_event.is_set():
        pkg = create_package()

        with packages_lock:
            packages.append(pkg)

        info(
            f"Создана | id={pkg['id']} | {pkg['origin']} -> {pkg['destination']} | "
            f"тип={pkg['shipment_type']}"
        )

        queue.put(pkg)
        current = adjust_delay(queue, delay)
        time.sleep(current)


def monitor(queue, stop_event, stats, stats_lock):
    while not stop_event.is_set():
        with stats_lock:
            info(

                f"Монитор | очередь={queue.qsize()} | обработано={stats['processed']} | "
                f"сумма={stats['total_cost']:.2f}"

            )

        time.sleep(2)


def manager(
    queue,
    workers,
    workers_lock,
    cost_function,
    stats,
    stats_lock,
    stop_event,
    max_workers,
):
    while not stop_event.is_set():
        if queue.qsize() > 10:
            with workers_lock:
                if len(workers) < max_workers:
                    name = f"Extra-{len(workers)}"
                    thread = threading.Thread(
                        target=worker,
                        args=(name, queue, cost_function, stats, stats_lock, stop_event),
                    )
                    thread.start()
                    workers.append(thread)
                    info(f"Добавлен воркер | имя={name}")

        time.sleep(3)


def serialize_packages(packages):
    prepared = []

    for package in packages:
        lat, lon = package["coords"]
        prepared.append(
            {
                "id": package["id"],
                "origin": package["origin"],
                "origin_country": package["origin_country"],
                "destination": package["destination"],
                "country": package["country"],
                "start_lat": package["start_coords"][0],
                "start_lon": package["start_coords"][1],
                "lat": lat,
                "lon": lon,
                "priority": package["priority"],
                "shipment_type": package["shipment_type"],
                "status": package["status"],
                "weight": package["weight"],
                "distance_km": package["distance_km"],
                "cost": package["cost"],
                "created_at": package["created_at"],
                "processed_at": package["processed_at"],
                "route": package["route"],
                "route_is_fallback": package["route_is_fallback"],
            }
        )

    return prepared


def get_map_center(packages):
    if not packages:
        return {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}

    lat_sum = 0
    lon_sum = 0
    points_count = 0

    for package in packages:
        start_lat, start_lon = package["start_coords"]
        end_lat, end_lon = package["coords"]

        if start_lat is not None and start_lon is not None:
            lat_sum += start_lat
            lon_sum += start_lon
            points_count += 1

        if end_lat is not None and end_lon is not None:
            lat_sum += end_lat
            lon_sum += end_lon
            points_count += 1

    if points_count == 0:
        return {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}

    return {
        "lat": lat_sum / points_count,
        "lon": lon_sum / points_count,
    }


def build_map_payload(packages, stats, simulation_active):
    return {
        "map_center": get_map_center(packages),
        "packages": serialize_packages(packages),
        "simulation_active": simulation_active,
        "stats": {
            "processed": stats["processed"],
            "total_cost": round(stats["total_cost"], 2),
            "total": len(packages),
        },
    }


def build_map_html(payload):
    template = MAP_TEMPLATE_FILE.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))


def get_state(packages, packages_lock, stats, stats_lock):
    with packages_lock:
        packages_temp = [package.copy() for package in packages]

    with stats_lock:
        stats_temp = dict(stats)

    return packages_temp, stats_temp


def publish_map_state(packages, stats, simulation_active):
    payload = build_map_payload(packages, stats, simulation_active)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_FILE.write_text(build_map_html(payload), encoding="utf-8")
    MAP_DATA_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def map_publisher(stop_event, packages, packages_lock, stats, stats_lock):
    while not stop_event.is_set():
        packages_temp, stats_temp = get_state(
            packages,
            packages_lock,
            stats,
            stats_lock,
        )
        publish_map_state(packages_temp, stats_temp, simulation_active=True)
        time.sleep(MAP_REFRESH_SECONDS)


def start_map_server():
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(BASE_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def run():
    sim_time = int(input("Время: "))
    workers_n = int(input("Воркеры: "))
    delay = float(input("Задержка: "))

    info(
        f"Старт симуляции | время={sim_time} сек | воркеры={workers_n} | задержка={delay:.2f} сек"
    )

    queue = Queue()
    stop_event = threading.Event()
    map_stop_event = threading.Event()

    stats = {
        "processed": 0,
        "total_cost": 0,
    }
    packages = []

    stats_lock = threading.Lock()
    workers_lock = threading.Lock()
    packages_lock = threading.Lock()

    server, server_thread = start_map_server()
    map_url = f"http://127.0.0.1:{server.server_port}/{MAP_URL_PATH}"

    initial_packages, initial_stats = get_state(packages, packages_lock, stats, stats_lock)
    publish_map_state(initial_packages, initial_stats, simulation_active=True)
    webbrowser.open(map_url)
    info(f"Карта запущена | url={map_url}")

    cost_function = create_cost_calculator(10)
    workers = []

    for i in range(workers_n):
        thread = threading.Thread(
            target=worker,
            args=(f"W{i}", queue, cost_function, stats, stats_lock, stop_event),
        )
        thread.start()
        workers.append(thread)

    generator_thread = threading.Thread(
        target=generate,
        args=(queue, stop_event, delay, packages, packages_lock),
    )
    monitor_thread = threading.Thread(
        target=monitor,
        args=(queue, stop_event, stats, stats_lock),
    )
    map_thread = threading.Thread(
        target=map_publisher,
        args=(map_stop_event, packages, packages_lock, stats, stats_lock),
        daemon=True,
    )
    manager_thread = threading.Thread(
        target=manager,
        args=(queue, workers, workers_lock, cost_function, stats, stats_lock, stop_event, workers_n + 3),
    )

    generator_thread.start()
    monitor_thread.start()
    map_thread.start()
    manager_thread.start()

    time.sleep(sim_time)
    stop_event.set()
    generator_thread.join()
    queue.join()

    with workers_lock:
        active_workers = list(workers)

    for _ in active_workers:
        queue.put(None)

    for worker_thread in active_workers:
        worker_thread.join()

    monitor_thread.join()
    manager_thread.join()

    final_packages, final_stats = get_state(packages, packages_lock, stats, stats_lock)
    publish_map_state(final_packages, final_stats, simulation_active=False)
    map_stop_event.set()
    map_thread.join(timeout=MAP_REFRESH_SECONDS + 1)
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=2)
    info(f"Карта сохранена | файл={MAP_FILE}")
    info("Симуляция завершена")

run()
