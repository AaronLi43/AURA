"""Central configuration for the AURA privacy rewriting pipeline."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
INPUT_JSONL = INPUT_DIR / "example_transcripts.jsonl"
DB_PATH = Path(os.getenv("NOBRANCH_DB_PATH", str(BASE_DIR / "pipeline.db")))
OUTPUT_DIR = Path(os.getenv("NOBRANCH_OUTPUT_DIR", str(BASE_DIR / "output")))

# ── OpenAI models ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_PROVIDER = os.getenv("NB_LLM_PROVIDER", "openai").strip().lower()
DISABLE_REASONING = os.getenv("NB_DISABLE_REASONING", "0").strip().lower() in {"1", "true", "yes", "on"}
EXCLUDE_REASONING = os.getenv("NB_EXCLUDE_REASONING", "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_OPENROUTER_CALLS = os.getenv("NB_DEBUG_OPENROUTER_CALLS", "0").strip().lower() in {"1", "true", "yes", "on"}
LENGTH_RERUN_ATTEMPTS = max(1, int(os.getenv("NB_LENGTH_RERUN_ATTEMPTS", "1")))

MASKER_MODEL = os.getenv("NB_MASKER_MODEL", OPENAI_MODEL)
REFILLER_MODEL = os.getenv("NB_REFILLER_MODEL", OPENAI_MODEL)
ATTACKER_MODEL = os.getenv("NB_ATTACKER_MODEL", OPENAI_MODEL)
KEEPER_MODEL = os.getenv("NB_KEEPER_MODEL", OPENAI_MODEL)
MODULATOR_MODEL = os.getenv("NB_MODULATOR_MODEL", OPENAI_MODEL)
INIT_MODEL = os.getenv("NB_INIT_MODEL", OPENAI_MODEL)

# ── Iteration settings ────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("NB_MAX_ITERATIONS", "5"))
VARIATIONS_PER_ROUND = int(os.getenv("NB_VARIATIONS_PER_ROUND", "4"))
MASKER_CONVERGE_ROUNDS = int(os.getenv("NB_MASKER_CONVERGE_ROUNDS", "5"))
JSON_MAX_TOKENS = int(os.getenv("NB_JSON_MAX_TOKENS", "16000"))
TEXT_MAX_TOKENS = int(os.getenv("NB_TEXT_MAX_TOKENS", "16000"))
MAX_TOO_SPECIFIC_ATTRS = int(os.getenv("NB_MAX_TOO_SPECIFIC_ATTRS", "2"))
CERTAINTY_THRESHOLD_FOR_BLACKLIST = int(
    os.getenv("NB_CERTAINTY_THRESHOLD", "3")
)

# ── Parallelism ───────────────────────────────────────────────────────
RUN_ALL_MAX_WORKERS = int(os.getenv("NB_RUN_ALL_MAX_WORKERS", "4"))
REFILLER_MAX_WORKERS = int(os.getenv("NB_REFILLER_MAX_WORKERS", "8"))
EVAL_MAX_WORKERS = int(os.getenv("NB_EVAL_MAX_WORKERS", "8"))


def strip_think_tags(text: str) -> str:
    """Remove optional reasoning tags from model output if present."""
    if text and "<think>" in text:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return text


def safe_json_loads(raw: str):
    """Tolerant JSON loader. Returns {} on unrecoverable error.

    Tries strict ``json.loads`` first; on failure, scans for the first
    balanced ``{...}`` or ``[...]`` segment and attempts to parse that.
    Used to recover from truncated or partially malformed model output.
    """
    if not raw:
        return {}
    text = strip_think_tags(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        start = text.find(start_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return {}


def _merge_extra_body(default_extra: dict, caller_extra: dict) -> dict:
    merged = dict(default_extra)
    for key, value in caller_extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def _reasoning_tokens_from_usage(usage) -> int | None:
    payload = _usage_to_dict(usage)
    details = payload.get("completion_tokens_details") or {}
    if isinstance(details, dict) and "reasoning_tokens" in details:
        return details.get("reasoning_tokens")
    return payload.get("reasoning_tokens")


def _debug_openrouter_response(response, attempt: int, max_attempts: int) -> None:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "") or ""
    finish_reason = getattr(choice, "finish_reason", None)
    usage_payload = _usage_to_dict(getattr(response, "usage", None))
    reasoning_tokens = _reasoning_tokens_from_usage(getattr(response, "usage", None))
    print(
        f"[openrouter-debug] attempt={attempt}/{max_attempts} "
        f"finish_reason={finish_reason} reasoning_tokens={reasoning_tokens}",
        flush=True,
    )
    print("[openrouter-debug] output_begin", flush=True)
    print(content, flush=True)
    print("[openrouter-debug] output_end", flush=True)
    print(
        "[openrouter-debug] usage="
        + json.dumps(usage_payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _wrap_client_with_extra_body(client: OpenAI, extra_body: dict) -> OpenAI:
    """Force every chat.completions.create call to include ``extra_body``.

    Existing ``extra_body`` kwargs win on key collision; this only fills in
    keys the caller did not already set.
    """
    if not extra_body and not DEBUG_OPENROUTER_CALLS and LENGTH_RERUN_ATTEMPTS <= 1:
        return client

    completions = client.chat.completions
    original_create = completions.create

    def _create_with_extra_body(*args, **kwargs):
        caller_extra = kwargs.pop("extra_body", None) or {}
        merged = _merge_extra_body(extra_body, caller_extra)
        if merged:
            kwargs["extra_body"] = merged
        max_attempts = max(1, LENGTH_RERUN_ATTEMPTS)
        for attempt in range(1, max_attempts + 1):
            response = original_create(*args, **kwargs)
            if DEBUG_OPENROUTER_CALLS:
                _debug_openrouter_response(response, attempt, max_attempts)
            choice = response.choices[0] if getattr(response, "choices", None) else None
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length" and attempt < max_attempts:
                print(
                    "[openrouter-debug] finish_reason=length; rerunning call",
                    flush=True,
                )
                continue
            return response
        return response

    completions.create = _create_with_extra_body
    return client


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _openrouter_reasoning_extra_body() -> dict:
    """Build OpenRouter reasoning controls.

    Qwen models on OpenRouter can spend a long time in a hidden reasoning
    phase unless reasoning is disabled.  ``run_openrouter_sample.py`` sets
    ``NB_DISABLE_REASONING=1`` explicitly; ``run_expanded_privacy.py`` did
    not, which made OpenRouter rewrite runs look much slower than GPT runs
    even when ``--only-base-attri`` skips the adaptive discovery steps.
    """
    extra_body: dict = {}
    if _env_flag("NB_EXCLUDE_REASONING"):
        extra_body["reasoning"] = {"exclude": True}

    disable_reasoning = _env_flag("NB_DISABLE_REASONING")
    if os.getenv("NB_DISABLE_REASONING") is None:
        model_name = os.getenv("NB_MASKER_MODEL", OPENAI_MODEL).lower()
        if "qwen" in model_name:
            disable_reasoning = True

    if disable_reasoning:
        extra_body.setdefault("reasoning", {})["enabled"] = False
    return extra_body


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def get_pipeline_client() -> OpenAI:
    if LLM_PROVIDER == "openai":
        return get_openai_client()
    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment/.env")
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        extra_body = _openrouter_reasoning_extra_body()
        client = _wrap_client_with_extra_body(client, extra_body)
        return client
    raise RuntimeError(f"Unsupported NB_LLM_PROVIDER: {LLM_PROVIDER}")
