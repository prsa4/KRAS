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
        "map_file": output_dir / "shipments_map.html",
        "map_data_file": output_dir / "shipments_data.json",
        "map_template_file": base_dir / "shipments_map_template.html",
        "map_url_path": (output_dir / "shipments_map.html").relative_to(base_dir).as_posix(),
        "map_refresh_seconds": MAP_REFRESH_SECONDS,
        "default_map_center": DEFAULT_MAP_CENTER,
        "osrm_route_url": OSRM_ROUTE_URL,
        "osrm_min_request_interval": OSRM_MIN_REQUEST_INTERVAL,
        "osrm_retry_attempts": OSRM_RETRY_ATTEMPTS,
        "country_pool": COUNTRY_POOL,
    }


def create_stats():
    return {
        "processed": 0,
        "total_cost": 0.0,
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


def make_package_factory(fake_service, rng, country_pool, timestamp_supplier):
    choose_route_points = retry_until_distinct(
        lambda: choose_destination(fake_service, rng, country_pool)
    )

    def create_package():
        origin, (start_lat, start_lon), destination, (end_lat, end_lon) = (
            choose_route_points()
        )
        priority = rng.choice([0, 1])

        return {
            "id": rng.randint(1000, 9999),
            "weight": round(rng.uniform(0.5, 10), 2),
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
            "created_at": timestamp_supplier(),
            "processed_at": None,
        }

    return create_package


def make_cost_calculator(base_price, rng):
    def calculate(weight, priority, distance):
        multiplier = rng.uniform(1.5, 3)

        if priority == 1:
            multiplier *= 1.5

        return base_price + weight * multiplier + distance * 0.5

    return calculate


def process_package(
    package,
    cost_function,
    stats,
    stats_lock,
    route_fetcher,
    rng,
    timestamp_supplier,
):
    start_lat, start_lon = package["start_coords"]
    lat, lon = package["coords"]

    if start_lat is None or start_lon is None or lat is None or lon is None:
        package["status"] = "geocode_failed"
        return

    distance = calculate_distance(start_lat, start_lon, lat, lon)
    route_data = route_fetcher(start_lat, start_lon, lat, lon)
    cost = cost_function(package["weight"], package["priority"], distance)

    package["distance_km"] = round(distance, 1)
    package["cost"] = round(cost, 2)
    package["route"] = route_data["points"]
    package["route_is_fallback"] = route_data["is_fallback"]
    package["status"] = "processed"
    package["processed_at"] = timestamp_supplier()

    info(
        f"Обработана | id={package['id']} | {package['origin']} -> {package['destination']} | "
        f"тип={package['shipment_type']} | расстояние={distance:.1f} км | стоимость={cost:.2f}"
    )

    time.sleep(rng.uniform(0.3, 1.0))

    with stats_lock:
        stats["processed"] += 1
        stats["total_cost"] += cost


def worker_loop(
    name,
    runtime,
    cost_function,
    route_fetcher,
    rng,
    timestamp_supplier,
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
            cost_function,
            runtime["stats"],
            runtime["stats_lock"],
            route_fetcher,
            rng,
            timestamp_supplier,
        )

        time.sleep(1 / speed)
        queue.task_done()

    info(f"{name} остановлен")


def adjust_delay(queue, base_delay):
    size = queue.qsize()

    if size > 15:
        return base_delay * 1.5
    if size < 5:
        return max(0.2, base_delay * 0.7)

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


def monitor_loop(runtime):
    queue = runtime["queue"]
    stop_event = runtime["stop_event"]

    while not stop_event.is_set():
        with runtime["stats_lock"]:
            info(
                f"Монитор | очередь={queue.qsize()} | обработано={runtime['stats']['processed']} | "
                f"сумма={runtime['stats']['total_cost']:.2f}"
            )

        time.sleep(2)


def start_worker(name, runtime, cost_function, route_fetcher, rng, timestamp_supplier):
    thread = threading.Thread(
        target=worker_loop,
        args=(
            name,
            runtime,
            cost_function,
            route_fetcher,
            rng,
            timestamp_supplier,
        ),
    )
    thread.start()
    return thread


def manager_loop(runtime, cost_function, route_fetcher, max_workers, rng, timestamp_supplier):
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
        "lat": package["coords"][0],
        "lon": package["coords"][1],
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


def serialize_packages(packages):
    return list(map(serialize_package, packages))


def iter_package_points(packages):
    for package in packages:
        yield package["start_coords"]
        yield package["coords"]


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
    return {
        "map_center": get_map_center(packages, default_center),
        "packages": serialize_packages(packages),
        "simulation_active": simulation_active,
        "stats": {
            "processed": stats["processed"],
            "total_cost": round(stats["total_cost"], 2),
            "total": len(packages),
        },
    }


def build_map_html(template_file, payload):
    template = template_file.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))


def snapshot_state(runtime):
    with runtime["packages_lock"]:
        packages_copy = [package.copy() for package in runtime["packages"]]

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
        "Старт симуляции | "
        f"время={settings['sim_time']} сек | воркеры={settings['workers_n']} | "
        f"задержка={settings['delay']:.2f} сек"
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