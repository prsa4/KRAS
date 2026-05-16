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


class CargoGeneratorService:
    def __init__(
        self,
        model_path=MODEL_PATH,
        device=MODEL_DEVICE,
        n_ctx=MODEL_N_CTX,
        n_threads=MODEL_N_THREADS,
        n_gpu_layers=MODEL_GPU_LAYERS,
        model_batch_size=MODEL_BATCH_SIZE,
        initial_buffer_size=INITIAL_CARGO_BUFFER_SIZE,
        buffer_low_watermark=BUFFER_LOW_WATERMARK,
        prefetch_idle_seconds=PREFETCH_IDLE_SECONDS,
        generation_attempts=MODEL_GENERATION_ATTEMPTS,
        model_max_tokens=MODEL_MAX_TOKENS,
        recent_history_size=RECENT_CARGO_HISTORY_SIZE,
        startup_ready_buffer_size=STARTUP_READY_BUFFER_SIZE,
        max_blocking_fill_rounds=MAX_BLOCKING_FILL_ROUNDS,
    ):
        self.model_path = model_path
        self.device = device if device in {"cpu", "gpu"} else "cpu"
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers if self.device == "gpu" else 0
        self.model_batch_size = model_batch_size
        self.initial_buffer_size = initial_buffer_size
        self.buffer_low_watermark = buffer_low_watermark
        self.prefetch_idle_seconds = prefetch_idle_seconds
        self.generation_attempts = generation_attempts
        self.model_max_tokens = model_max_tokens
        self.recent_history_size = recent_history_size
        self.startup_ready_buffer_size = startup_ready_buffer_size
        self.max_blocking_fill_rounds = max_blocking_fill_rounds

        self._model_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._fill_lock = threading.Lock()
        self._history_lock = threading.Lock()

        self._llm = None
        self._load_error = None
        self._cargo_buffer = []
        self._recent_cargo_signatures = deque()
        self._recent_cargo_signature_set = set()

    def get_runtime_config(self):
        return {
            "model_path": self.model_path,
            "device": self.device,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "chat_format": "gemma",
        }

    def _build_llama_options(self):
        runtime = self.get_runtime_config()
        return {
            "model_path": runtime["model_path"],
            "n_ctx": runtime["n_ctx"],
            "n_threads": runtime["n_threads"],
            "n_gpu_layers": runtime["n_gpu_layers"],
            "verbose": False,
            "chat_format": runtime["chat_format"],
        }

    def _load_generator_model(self):
        if self._llm is not None:
            return self._llm

        with self._model_lock:
            if self._llm is not None:
                return self._llm

            if self._load_error is not None:
                raise RuntimeError("Cargo model is unavailable") from self._load_error

            try:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Gemma model not found: {self.model_path}")

                llm = Llama(**self._build_llama_options())
            except Exception as exc:
                self._load_error = exc
                raise RuntimeError("Cargo model failed to load") from exc

            self._llm = llm
            return llm

    def _build_cargo_prompt(self, batch_size):
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

    def _generate_model_text(self, prompt):
        llm = self._load_generator_model()

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
            max_tokens=self.model_max_tokens,
            top_p=0.95,
            repeat_penalty=1.14,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _extract_json_payload(self, text):
        match = re.search(r"\[.*\]", text, re.DOTALL)

        if not match:
            match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("Model response does not contain JSON")

        return json.loads(match.group(0))

    def _to_positive_int(self, value, default):
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default

        return max(1, min(20, result))

    def _to_positive_float(self, value, default):
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default

        return round(max(0.1, min(80.0, result)), 2)

    def _normalize_text(self, value, default):
        text = str(value or "").strip()
        return text or default

    def _normalize_brand(self, value):
        brand = self._normalize_text(value, "Generated")
        brand = re.sub(r"\s+", " ", brand)
        return brand[:40]

    def _normalize_unit(self, value):
        unit = self._normalize_text(value, "кор.").lower()

        if unit in {"unit", "units", "piece", "pieces", "pcs", "pc", "шт", "шт."}:
            return "шт."

        if unit in {"box", "boxes", "carton", "cartons", "кор", "кор."}:
            return "кор."

        if unit in {"pallet", "pallets", "пал", "пал."}:
            return "пал."

        return "кор."

    def _normalize_model_name(self, value, default):
        model_name = self._normalize_text(value, default)
        model_name = re.sub(r"\s+", " ", model_name)
        return model_name[:60]

    def _build_cargo_summary(self, cargo):
        return (
            f"{cargo['brand']} {cargo['name']} {cargo['model']} | "
            f"{cargo['quantity']} {cargo['unit']}"
        )

    def _contains_forbidden_cargo_terms(self, *parts):
        haystack = " ".join(str(part or "").lower() for part in parts)
        return any(term in haystack for term in FORBIDDEN_CARGO_TERMS)

    def _looks_like_business_entity(self, *parts):
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

    def _cargo_signature(self, candidate):
        return (
            candidate.get("brand", "").strip().lower(),
            candidate.get("name", "").strip().lower(),
            candidate.get("model", "").strip().lower(),
        )

    def _get_blocked_cargo_signatures(self):
        with self._buffer_lock:
            buffer_signatures = {self._cargo_signature(item) for item in self._cargo_buffer}

        with self._history_lock:
            recent_signatures = set(self._recent_cargo_signature_set)

        return buffer_signatures | recent_signatures

    def _remember_cargo_signatures(self, cargos):
        with self._history_lock:
            for cargo in cargos:
                signature = self._cargo_signature(cargo)

                if signature in self._recent_cargo_signature_set:
                    continue

                self._recent_cargo_signatures.append(signature)
                self._recent_cargo_signature_set.add(signature)

                while len(self._recent_cargo_signatures) > self.recent_history_size:
                    expired = self._recent_cargo_signatures.popleft()
                    self._recent_cargo_signature_set.discard(expired)

    def _is_distinct_cargo(self, candidate, seen_signatures, blocked_signatures=None):
        signature = self._cargo_signature(candidate)

        if blocked_signatures is not None and signature in blocked_signatures:
            return False

        if signature in seen_signatures:
            return False

        seen_signatures.add(signature)
        return True

    def _normalize_cargo_payload(self, payload, rng=None):
        rng = rng or random

        quantity = self._to_positive_int(
            payload.get("quantity"),
            rng.randint(1, 18),
        )

        estimated_weight_kg = self._to_positive_float(
            payload.get("estimated_weight_kg"),
            quantity * rng.uniform(0.4, 2.8),
        )

        cargo = {
            "brand": self._normalize_brand(payload.get("brand")),
            "name": self._normalize_text(payload.get("name"), "Случайный товар"),
            "model": self._normalize_model_name(payload.get("model"), "Series X"),
            "unit": self._normalize_unit(payload.get("unit")),
            "quantity": quantity,
            "estimated_weight_kg": estimated_weight_kg,
        }

        cargo["summary"] = self._build_cargo_summary(cargo)
        return cargo

    def _fallback_cargo(self, rng=None):
        rng = rng or random

        cargo = {
            "brand": f"Generated-{rng.randint(100, 999)}",
            "name": f"Товар {rng.randint(1000, 9999)}",
            "model": f"Series-{rng.randint(10, 99)}",
            "unit": "кор.",
            "quantity": rng.randint(1, 20),
            "estimated_weight_kg": round(rng.uniform(1.0, 30.0), 2),
        }

        cargo["summary"] = self._build_cargo_summary(cargo)
        return cargo

    def _is_valid_cargo_payload(self, payload):
        if not isinstance(payload, dict):
            return False

        brand = payload.get("brand")
        name = payload.get("name")
        model_name = payload.get("model")
        unit = str(payload.get("unit") or "").strip().lower()

        if not brand or not name or not model_name:
            return False

        if self._contains_forbidden_cargo_terms(brand, name, model_name):
            return False

        if self._looks_like_business_entity(brand, name, model_name):
            return False

        if unit not in ALLOWED_UNITS:
            return False

        if len(str(name).strip()) < 3 or len(str(model_name).strip()) < 2:
            return False

        return True

    def _generate_cargo_batch_from_model(self, batch_size, rng=None):
        rng = rng or random
        last_error = None
        cargos = []
        seen_signatures = set()
        blocked_signatures = self._get_blocked_cargo_signatures()

        for _ in range(self.generation_attempts):
            try:
                missing = max(1, batch_size - len(cargos))
                raw_text = self._generate_model_text(self._build_cargo_prompt(missing))
                payload = self._extract_json_payload(raw_text)

                if isinstance(payload, dict):
                    payload = [payload]

                if not isinstance(payload, list) or not payload:
                    continue

                for item in payload:
                    if not self._is_valid_cargo_payload(item):
                        continue

                    normalized = self._normalize_cargo_payload(item, rng)

                    if not self._is_distinct_cargo(
                        normalized,
                        seen_signatures,
                        blocked_signatures,
                    ):
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

    def _fill_cargo_buffer(self, rng=None):
        rng = rng or random

        with self._fill_lock:
            try:
                cargos = self._generate_cargo_batch_from_model(self.model_batch_size, rng)
            except Exception:
                if self._load_error is not None:
                    cargos = [self._fallback_cargo(rng) for _ in range(self.model_batch_size)]
                else:
                    cargos = []

            if cargos:
                self._remember_cargo_signatures(cargos)
                with self._buffer_lock:
                    self._cargo_buffer.extend(cargos)

            return len(cargos)

    def get_cargo_buffer_size(self):
        with self._buffer_lock:
            return len(self._cargo_buffer)

    def prefill_cargo_buffer(self, target_size=None, rng=None, max_rounds=None):
        rng = rng or random
        target_size = target_size or self.initial_buffer_size
        stalled_attempts = 0
        rounds = 0

        while self.get_cargo_buffer_size() < target_size:
            if max_rounds is not None and rounds >= max_rounds:
                break

            added = self._fill_cargo_buffer(rng)
            rounds += 1

            if added > 0:
                stalled_attempts = 0
                continue

            stalled_attempts += 1

            if stalled_attempts >= self.generation_attempts:
                break

        return self.get_cargo_buffer_size()

    def cargo_prefetch_loop(
        self,
        stop_event,
        target_size=None,
        low_watermark=None,
        idle_seconds=None,
        rng=None,
    ):
        rng = rng or random
        target_size = target_size or self.initial_buffer_size
        low_watermark = (
            self.buffer_low_watermark if low_watermark is None else low_watermark
        )
        idle_seconds = self.prefetch_idle_seconds if idle_seconds is None else idle_seconds

        while not stop_event.is_set():
            if self.get_cargo_buffer_size() <= low_watermark:
                self.prefill_cargo_buffer(target_size, rng)

            stop_event.wait(idle_seconds)

    def generate_random_cargo(self, rng=None):
        rng = rng or random

        with self._buffer_lock:
            if self._cargo_buffer:
                return self._cargo_buffer.pop(0)

        self.prefill_cargo_buffer(
            self.startup_ready_buffer_size,
            rng,
            max_rounds=self.max_blocking_fill_rounds,
        )

        with self._buffer_lock:
            if self._cargo_buffer:
                return self._cargo_buffer.pop(0)

        return self._fallback_cargo(rng)

    def generate_cargo_batch(self, size, rng=None):
        rng = rng or random
        return [self.generate_random_cargo(rng) for _ in range(size)]

    def preview_cargo_batch(self, size=3):
        return self.generate_cargo_batch(size)


def main():
    service = CargoGeneratorService()
    for cargo in service.preview_cargo_batch():
        print(json.dumps(cargo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()