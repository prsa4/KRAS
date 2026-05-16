import json
import os
import random
import re
import threading
from collections import deque
from pathlib import Path

from llama_cpp import Llama


MODEL_PATH = str(
    Path(__file__).resolve().parent / "models" / "Gemma3-UNCENSORED-1B.Q4_K_M.gguf"
)
MODEL_DEVICE = os.getenv("CARGO_MODEL_DEVICE", "cpu").strip().lower()


def _read_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MODEL_N_CTX = _read_int_env("CARGO_MODEL_N_CTX", 2048)
MODEL_N_THREADS = _read_int_env("CARGO_MODEL_N_THREADS", 0)
MODEL_GPU_LAYERS = _read_int_env(
    "CARGO_MODEL_GPU_LAYERS",
    -1 if MODEL_DEVICE == "gpu" else 0,
)

MODEL_BATCH_SIZE = 4
INITIAL_CARGO_BUFFER_SIZE = 12
BUFFER_LOW_WATERMARK = 8
PREFETCH_IDLE_SECONDS = 0.12
MODEL_GENERATION_ATTEMPTS = 3
MODEL_MAX_TOKENS = 384
RECENT_CARGO_HISTORY_SIZE = 256
STARTUP_READY_BUFFER_SIZE = 1
MAX_BLOCKING_FILL_ROUNDS = 1

FORBIDDEN_CARGO_TERMS = {
    "завод",
    "фабрика",
    "предприятие",
    "склад",
    "цех",
    "комбинат",
    "буровая",
    "электростанция",
    "станция",
    "депо",
    "терминал",
    "хаб",
    "warehouse",
    "factory",
    "plant",
    "facility",
    "building",
    "infrastructure",
    "terminal",
    "hub",
    "depot",
    "компания",
    "магазин",
    "сервис",
    "супермаркет",
    "бренд",
    "дистрибьютор",
    "company",
    "store",
    "service",
    "market",
    "distributor",
}

ALLOWED_UNITS = {
    "шт.",
    "шт",
    "кор.",
    "кор",
    "пал.",
    "пал",
    "unit",
    "units",
    "piece",
    "pieces",
    "pcs",
    "pc",
    "box",
    "boxes",
    "carton",
    "cartons",
    "pallet",
    "pallets",
}

_MODEL_LOCK = threading.Lock()
_BUFFER_LOCK = threading.Lock()
_FILL_LOCK = threading.Lock()
_HISTORY_LOCK = threading.Lock()

_MODEL_STATE = {
    "llm": None,
    "load_error": None,
}

_CARGO_BUFFER = []
_RECENT_CARGO_SIGNATURES = deque()
_RECENT_CARGO_SIGNATURE_SET = set()


def get_model_runtime_config():
    device = MODEL_DEVICE if MODEL_DEVICE in {"cpu", "gpu"} else "cpu"
    return {
        "model_path": MODEL_PATH,
        "device": device,
        "n_ctx": MODEL_N_CTX,
        "n_threads": MODEL_N_THREADS,
        "n_gpu_layers": MODEL_GPU_LAYERS if device == "gpu" else 0,
        "chat_format": "gemma",
    }


def _build_llama_options():
    runtime = get_model_runtime_config()
    return {
        "model_path": runtime["model_path"],
        "n_ctx": runtime["n_ctx"],
        "n_threads": runtime["n_threads"],
        "n_gpu_layers": runtime["n_gpu_layers"],
        "verbose": False,
        "chat_format": runtime["chat_format"],
    }


def _load_generator_model():
    if _MODEL_STATE["llm"] is not None:
        return _MODEL_STATE["llm"]

    with _MODEL_LOCK:
        if _MODEL_STATE["llm"] is not None:
            return _MODEL_STATE["llm"]

        if _MODEL_STATE["load_error"] is not None:
            raise RuntimeError("Cargo model is unavailable") from _MODEL_STATE["load_error"]

        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Gemma model not found: {MODEL_PATH}")

            llm = Llama(**_build_llama_options())
        except Exception as exc:
            _MODEL_STATE["load_error"] = exc
            raise RuntimeError("Cargo model failed to load") from exc

        _MODEL_STATE["llm"] = llm
        return llm


def _build_cargo_prompt(batch_size):
    return (
        f"Верни JSON-массив ровно из {batch_size} разных товаров для международной доставки. "
        "Каждый объект должен быть именно товаром, устройством, аксессуаром, инструментом или партией продукции. "
        "Сделай товары максимально разными по типу и брендам, не повторяй один и тот же товар разными вариантами. "
        "Желательно смешивать категории: электроника, одежда, бытовые товары, инструменты, спорттовары, косметика, игрушки, автоаксессуары, офисные товары. "
        "Нельзя генерировать заводы, компании, магазины, склады, сервисы, здания и инфраструктуру. "
        "Поля строго такие: brand, name, model, unit, quantity, estimated_weight_kg. "
        "unit только из списка: шт., кор., пал. "
        "quantity от 1 до 20. estimated_weight_kg от 0.1 до 80.0. "
        "Верни только JSON-массив без пояснений."
    )


def _generate_model_text(prompt):
    llm = _load_generator_model()

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Ты генератор тестовых данных для логистической системы. "
                           "На русском пиши."
                           "Всегда отвечай только чистым JSON-массивом. "
                           "Не пиши объяснения, не используй markdown. "
                           "Генерируй только отдельные товары и партии товаров, а не компании, здания или инфраструктуру. "
                           "Старайся давать разнообразные бренды, модели и типы товаров без повторов."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.72,
        max_tokens=MODEL_MAX_TOKENS,
        top_p=0.95,
        repeat_penalty=1.14,
    )
    return response["choices"][0]["message"]["content"].strip()

def _extract_json_payload(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)

    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("Model response does not contain JSON")

    return json.loads(match.group(0))


def _to_positive_int(value, default):
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default

    return max(1, min(20, result))


def _to_positive_float(value, default):
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default

    return round(max(0.1, min(80.0, result)), 2)


def _normalize_text(value, default):
    text = str(value or "").strip()
    return text or default


def _normalize_brand(value):
    brand = _normalize_text(value, "Generated")
    brand = re.sub(r"\s+", " ", brand)
    return brand[:40]


def _normalize_unit(value):
    unit = _normalize_text(value, "кор.").lower()

    if unit in {"unit", "units", "piece", "pieces", "pcs", "pc", "шт", "шт."}:
        return "шт."

    if unit in {"box", "boxes", "carton", "cartons", "кор", "кор."}:
        return "кор."

    if unit in {"pallet", "pallets", "пал", "пал."}:
        return "пал."

    return "кор."


def _normalize_model_name(value, default):
    model_name = _normalize_text(value, default)
    model_name = re.sub(r"\s+", " ", model_name)
    return model_name[:60]


def _build_cargo_summary(cargo):
    return (
        f"{cargo['brand']} {cargo['name']} {cargo['model']} | "
        f"{cargo['quantity']} {cargo['unit']}"
    )


def _contains_forbidden_cargo_terms(*parts):
    haystack = " ".join(str(part or "").lower() for part in parts)
    return any(term in haystack for term in FORBIDDEN_CARGO_TERMS)


def _looks_like_business_entity(*parts):
    haystack = " ".join(str(part or "").lower() for part in parts)
    patterns = [
        r"\bооо\b",
        r"\bзао\b",
        r"\bоао\b",
        r"\bип\b",
        r"\bltd\b",
        r"\binc\b",
        r"\bcorp\b",
        r"\bllc\b",
    ]
    return any(re.search(pattern, haystack) for pattern in patterns)


def _cargo_signature(candidate):
    return (
        candidate.get("brand", "").strip().lower(),
        candidate.get("name", "").strip().lower(),
        candidate.get("model", "").strip().lower(),
    )


def _get_blocked_cargo_signatures():
    with _BUFFER_LOCK:
        buffer_signatures = {_cargo_signature(item) for item in _CARGO_BUFFER}

    with _HISTORY_LOCK:
        recent_signatures = set(_RECENT_CARGO_SIGNATURE_SET)

    return buffer_signatures | recent_signatures


def _remember_cargo_signatures(cargos):
    with _HISTORY_LOCK:
        for cargo in cargos:
            signature = _cargo_signature(cargo)

            if signature in _RECENT_CARGO_SIGNATURE_SET:
                continue

            _RECENT_CARGO_SIGNATURES.append(signature)
            _RECENT_CARGO_SIGNATURE_SET.add(signature)

            while len(_RECENT_CARGO_SIGNATURES) > RECENT_CARGO_HISTORY_SIZE:
                expired = _RECENT_CARGO_SIGNATURES.popleft()
                _RECENT_CARGO_SIGNATURE_SET.discard(expired)


def _is_distinct_cargo(candidate, seen_signatures, blocked_signatures=None):
    signature = _cargo_signature(candidate)

    if blocked_signatures is not None and signature in blocked_signatures:
        return False

    if signature in seen_signatures:
        return False

    seen_signatures.add(signature)
    return True


def _normalize_cargo_payload(payload, rng=None):
    rng = rng or random

    quantity = _to_positive_int(
        payload.get("quantity"),
        rng.randint(1, 18),
    )

    estimated_weight_kg = _to_positive_float(
        payload.get("estimated_weight_kg"),
        quantity * rng.uniform(0.4, 2.8),
    )

    cargo = {
        "brand": _normalize_brand(payload.get("brand")),
        "name": _normalize_text(payload.get("name"), "Случайный товар"),
        "model": _normalize_model_name(payload.get("model"), "Series X"),
        "unit": _normalize_unit(payload.get("unit")),
        "quantity": quantity,
        "estimated_weight_kg": estimated_weight_kg,
    }

    cargo["summary"] = _build_cargo_summary(cargo)
    return cargo


def _fallback_cargo(rng=None):
    rng = rng or random

    cargo = {
        "brand": f"Generated-{rng.randint(100, 999)}",
        "name": f"Товар {rng.randint(1000, 9999)}",
        "model": f"Series-{rng.randint(10, 99)}",
        "unit": "кор.",
        "quantity": rng.randint(1, 20),
        "estimated_weight_kg": round(rng.uniform(1.0, 30.0), 2),
    }

    cargo["summary"] = _build_cargo_summary(cargo)
    return cargo


def _is_valid_cargo_payload(payload):
    if not isinstance(payload, dict):
        return False

    brand = payload.get("brand")
    name = payload.get("name")
    model_name = payload.get("model")
    unit = str(payload.get("unit") or "").strip().lower()

    if not brand or not name or not model_name:
        return False

    if _contains_forbidden_cargo_terms(brand, name, model_name):
        return False

    if _looks_like_business_entity(brand, name, model_name):
        return False

    if unit not in ALLOWED_UNITS:
        return False

    if len(str(name).strip()) < 3 or len(str(model_name).strip()) < 2:
        return False

    return True


def _generate_cargo_batch_from_model(batch_size, rng=None):
    rng = rng or random
    last_error = None
    cargos = []
    seen_signatures = set()
    blocked_signatures = _get_blocked_cargo_signatures()

    for _ in range(MODEL_GENERATION_ATTEMPTS):
        try:
            missing = max(1, batch_size - len(cargos))
            raw_text = _generate_model_text(_build_cargo_prompt(missing))
            payload = _extract_json_payload(raw_text)

            if isinstance(payload, dict):
                payload = [payload]

            if not isinstance(payload, list) or not payload:
                continue

            for item in payload:
                if not _is_valid_cargo_payload(item):
                    continue

                normalized = _normalize_cargo_payload(item, rng)

                if not _is_distinct_cargo(normalized, seen_signatures, blocked_signatures):
                    continue

                cargos.append(normalized)

                if len(cargos) >= batch_size:
                    break

            if len(cargos) >= batch_size:
                return cargos

        except Exception as exc:
            last_error = exc

    if cargos:
        return cargos

    raise ValueError("Model cargo list does not contain valid product objects") from last_error


def _fill_cargo_buffer(rng=None):
    rng = rng or random

    with _FILL_LOCK:
        try:
            cargos = _generate_cargo_batch_from_model(MODEL_BATCH_SIZE, rng)
        except Exception:
            if _MODEL_STATE["load_error"] is not None:
                cargos = [_fallback_cargo(rng) for _ in range(MODEL_BATCH_SIZE)]
            else:
                cargos = []

        if cargos:
            _remember_cargo_signatures(cargos)
            with _BUFFER_LOCK:
                _CARGO_BUFFER.extend(cargos)

        return len(cargos)


def get_cargo_buffer_size():
    with _BUFFER_LOCK:
        return len(_CARGO_BUFFER)


def prefill_cargo_buffer(target_size=INITIAL_CARGO_BUFFER_SIZE, rng=None, max_rounds=None):
    rng = rng or random
    stalled_attempts = 0
    rounds = 0

    while get_cargo_buffer_size() < target_size:
        if max_rounds is not None and rounds >= max_rounds:
            break

        added = _fill_cargo_buffer(rng)
        rounds += 1

        if added > 0:
            stalled_attempts = 0
            continue

        stalled_attempts += 1

        if stalled_attempts >= MODEL_GENERATION_ATTEMPTS:
            break

    return get_cargo_buffer_size()


def cargo_prefetch_loop(
    stop_event,
    target_size=INITIAL_CARGO_BUFFER_SIZE,
    low_watermark=BUFFER_LOW_WATERMARK,
    idle_seconds=PREFETCH_IDLE_SECONDS,
    rng=None,
):
    rng = rng or random

    while not stop_event.is_set():
        if get_cargo_buffer_size() <= low_watermark:
            prefill_cargo_buffer(target_size, rng)

        stop_event.wait(idle_seconds)


def generate_random_cargo(rng=None):
    rng = rng or random

    with _BUFFER_LOCK:
        if _CARGO_BUFFER:
            return _CARGO_BUFFER.pop(0)

    prefill_cargo_buffer(STARTUP_READY_BUFFER_SIZE, rng, max_rounds=MAX_BLOCKING_FILL_ROUNDS)

    with _BUFFER_LOCK:
        if _CARGO_BUFFER:
            return _CARGO_BUFFER.pop(0)

    return _fallback_cargo(rng)


def generate_cargo_batch(size, rng=None):
    rng = rng or random
    return [generate_random_cargo(rng) for _ in range(size)]


def preview_cargo_batch(size=3):
    return generate_cargo_batch(size)


if __name__ == "__main__":
    for cargo in preview_cargo_batch():
        print(json.dumps(cargo, ensure_ascii=False, indent=2))