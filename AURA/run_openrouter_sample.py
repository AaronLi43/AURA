#!/usr/bin/env python3
"""Run one AURA pipeline sample through OpenRouter."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = BASE_DIR / "input" / "example_transcripts.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "openrouter_sample"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_SMOKE_PROMPT = "Reply with exactly: openrouter smoke test ok"
DEFAULT_FALLBACK_MAX_TOKENS = 6000
MODEL_MAX_TOKENS_DEFAULTS = (
    ("qwen", 81920),
    ("llama", 4096),
)
PHASE_MODEL_ENV_VARS = (
    "NB_INIT_MODEL",
    "NB_MASKER_MODEL",
    "NB_REFILLER_MODEL",
    "NB_ATTACKER_MODEL",
    "NB_KEEPER_MODEL",
    "NB_MODULATOR_MODEL",
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return slug.strip("_") or "sample"


def _output_prefix(model: str, transcript_id: str) -> str:
    return f"openrouter_{_slug(model)}_{_slug(transcript_id)}"


def _default_max_tokens_for_model(model: str) -> int:
    name = (model or "").lower()
    for needle, value in MODEL_MAX_TOKENS_DEFAULTS:
        if needle in name:
            return value
    return DEFAULT_FALLBACK_MAX_TOKENS


def _load_jsonl_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            transcript_id = str(record.get("conversation_id", "")).strip()
            text = str(record.get("user_message", "")).strip()
            if not transcript_id or not text:
                continue
            rows.append({"transcript_id": transcript_id, "text": text})
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _select_row(
    rows: list[dict[str, str]],
    transcript_id: str | None,
    pick: str,
) -> dict[str, str]:
    if transcript_id:
        for row in rows:
            if str(row.get("transcript_id", "")).strip().lower() == transcript_id.lower():
                return row
        raise RuntimeError(f"Transcript ID not found: {transcript_id}")
    if pick == "shortest":
        return min(rows, key=lambda row: len(str(row.get("text", ""))))
    return rows[0]


def _parse_id_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _sample_output_paths(output_dir: Path, model: str, transcript_id: str) -> tuple[Path, Path]:
    prefix = _output_prefix(model, transcript_id)
    return output_dir / f"{prefix}_output.csv", output_dir / f"{prefix}_output.json"


def _configure_openrouter(args: argparse.Namespace, db_path: Path) -> None:
    os.environ["NB_LLM_PROVIDER"] = "openrouter"
    os.environ["OPENAI_MODEL"] = args.model
    os.environ["NOBRANCH_DB_PATH"] = str(db_path)
    os.environ["NB_MASKER_CONVERGE_ROUNDS"] = str(args.masker_rounds)
    os.environ["NB_VARIATIONS_PER_ROUND"] = str(args.variations)
    os.environ["NB_REFILLER_MAX_WORKERS"] = str(args.refiller_workers)
    os.environ["NB_EVAL_MAX_WORKERS"] = str(args.eval_workers)
    os.environ["NB_JSON_MAX_TOKENS"] = str(args.json_max_tokens)
    os.environ["NB_TEXT_MAX_TOKENS"] = str(args.text_max_tokens)
    os.environ["NB_DISABLE_REASONING"] = "1" if args.disable_reasoning else "0"
    for env_name in PHASE_MODEL_ENV_VARS:
        os.environ[env_name] = args.model


def _reset_db(db_path: Path) -> None:
    for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        if path.exists():
            path.unlink()


def _write_jsonl_input(path: Path, transcript_id: str, text: str) -> None:
    payload = {"conversation_id": transcript_id, "user_message": text}
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_result(db_path: Path, transcript_id: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT document_id, original_text, final_text, final_privacy_score, status
        FROM documents
        WHERE document_id=?
        """,
        (transcript_id,),
    ).fetchone()
    iterations = conn.execute("SELECT count(*) FROM iterations").fetchone()[0]
    conn.close()
    if row is None:
        raise RuntimeError(f"No result row found for {transcript_id}")
    result = dict(row)
    result["iterations"] = iterations
    return result


def _write_outputs(csv_path: Path, json_path: Path, result: dict) -> None:
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "transcript_id",
                "provider",
                "model",
                "status",
                "final_privacy_score",
                "iterations",
                "final_text",
            ]
        )
        writer.writerow(
            [
                result.get("document_id", ""),
                result.get("provider", ""),
                result.get("model", ""),
                result.get("status", ""),
                result.get("final_privacy_score", ""),
                result.get("iterations", ""),
                result.get("final_text", ""),
            ]
        )


def _run_smoke_test(args: argparse.Namespace) -> int:
    smoke_db_path = args.output_dir / f"openrouter_{_slug(args.model)}_smoke.db"
    _configure_openrouter(args, smoke_db_path)

    import pipeline_config as cfg

    client = cfg.get_pipeline_client()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a concise API smoke-test assistant."},
            {"role": "user", "content": args.smoke_prompt},
        ],
        temperature=0,
        max_tokens=args.smoke_max_tokens,
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    if not content:
        raise RuntimeError("OpenRouter returned an empty message content")

    output_json = args.output_dir / f"openrouter_{_slug(args.model)}_smoke_output.json"
    payload = {
        "provider": "openrouter",
        "model": args.model,
        "prompt": args.smoke_prompt,
        "content": content,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(content)
    print(f"[openrouter] wrote {output_json}")
    return 0


def _read_sample_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _write_batch_summary(
    output_dir: Path,
    model: str,
    rows: list[dict],
) -> None:
    summary_json = output_dir / "batch_summary.json"
    summary_csv = output_dir / "batch_summary.csv"
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "transcript_id",
                "batch_status",
                "returncode",
                "output_json_exists",
                "status",
                "final_privacy_score",
                "final_text",
            ]
        )
        for row in rows:
            sample_csv, _ = _sample_output_paths(output_dir, model, row["transcript_id"])
            sample = _read_sample_csv(sample_csv)
            writer.writerow(
                [
                    row["transcript_id"],
                    row["batch_status"],
                    row.get("returncode", ""),
                    row["output_json_exists"],
                    sample.get("status", ""),
                    sample.get("final_privacy_score", ""),
                    sample.get("final_text", ""),
                ]
            )
    print(f"[batch] wrote {summary_json}")
    print(f"[batch] wrote {summary_csv}")


def _write_aggregate_csv(
    input_rows: list[dict[str, str]],
    output_dir: Path,
    model: str,
    aggregate_csv: Path,
) -> None:
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transcript_id", "text"])
        for row in input_rows:
            transcript_id = str(row.get("transcript_id", "")).strip()
            if not transcript_id:
                continue
            _, output_json = _sample_output_paths(output_dir, model, transcript_id)
            if not output_json.exists():
                continue
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                continue
            writer.writerow([transcript_id, payload.get("final_text", "")])
    print(f"[batch] wrote aggregate {aggregate_csv}")


def _build_sample_command(args: argparse.Namespace, transcript_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--input-jsonl",
        str(args.input_jsonl),
        "--transcript-id",
        transcript_id,
        "--model",
        args.model,
        "--output-dir",
        str(args.output_dir),
        "--masker-rounds",
        str(args.masker_rounds),
        "--variations",
        str(args.variations),
        "--refiller-workers",
        str(args.refiller_workers),
        "--eval-workers",
        str(args.eval_workers),
        "--json-max-tokens",
        str(args.json_max_tokens),
        "--text-max-tokens",
        str(args.text_max_tokens),
    ] + (["--disable-reasoning"] if args.disable_reasoning else [])


def _run_batch_worker(args: argparse.Namespace, transcript_id: str, log_dir: Path) -> dict:
    start = time.monotonic()
    log_path = log_dir / f"{_slug(transcript_id)}.log"
    cmd = _build_sample_command(args, transcript_id)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n=== STDOUT/STDERR ===\n")
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_s = round(time.monotonic() - start, 3)
    _, output_json = _sample_output_paths(args.output_dir, args.model, transcript_id)
    return {
        "transcript_id": transcript_id,
        "batch_status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "output_json_exists": output_json.exists(),
        "elapsed_s": elapsed_s,
        "log_path": str(log_path),
    }


def _run_batch(args: argparse.Namespace) -> int:
    rows = _load_jsonl_rows(args.input_jsonl)
    skip_ids = _parse_id_set(args.skip_ids)
    target_ids: list[str] = []
    summary_rows: list[dict] = []

    for row in rows:
        transcript_id = str(row.get("transcript_id", "")).strip()
        if not transcript_id:
            continue
        if transcript_id.lower() in skip_ids:
            summary_rows.append(
                {
                    "transcript_id": transcript_id,
                    "batch_status": "skipped_by_id",
                    "returncode": None,
                    "output_json_exists": False,
                    "elapsed_s": 0,
                    "log_path": None,
                }
            )
            continue
        _, output_json = _sample_output_paths(args.output_dir, args.model, transcript_id)
        if args.skip_completed and output_json.exists():
            summary_rows.append(
                {
                    "transcript_id": transcript_id,
                    "batch_status": "skipped_completed",
                    "returncode": 0,
                    "output_json_exists": True,
                    "elapsed_s": 0,
                    "log_path": None,
                }
            )
            continue
        target_ids.append(transcript_id)

    log_dir = args.output_dir / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[batch] total_rows={len(rows)} pending={len(target_ids)} "
        f"max_parallel={args.max_parallel}"
    )

    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {
            executor.submit(_run_batch_worker, args, transcript_id, log_dir): transcript_id
            for transcript_id in target_ids
        }
        pending = set(futures)
        completed_count = 0
        last_status_time = 0.0
        while pending:
            done, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if not done:
                if now - last_status_time >= 60:
                    running_ids = [futures[fut] for fut in pending]
                    preview = ", ".join(running_ids[:10])
                    if len(running_ids) > 10:
                        preview += f", ... (+{len(running_ids) - 10} more)"
                    print(
                        f"[batch] progress completed={completed_count}/{len(target_ids)} "
                        f"running={len(running_ids)} waiting_on=[{preview}]",
                        flush=True,
                    )
                    last_status_time = now
                continue
            for fut in done:
                completed_count += 1
                transcript_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    log_path = log_dir / f"{_slug(transcript_id)}.log"
                    log_path.write_text(f"Batch worker failed: {type(exc).__name__}: {exc}\n")
                    result = {
                        "transcript_id": transcript_id,
                        "batch_status": "failed",
                        "returncode": None,
                        "output_json_exists": False,
                        "elapsed_s": 0,
                        "log_path": str(log_path),
                    }
                summary_rows.append(result)
                print(
                    f"[batch] {result['batch_status']} {transcript_id} "
                    f"returncode={result['returncode']} elapsed_s={result['elapsed_s']} "
                    f"progress={completed_count}/{len(target_ids)}",
                    flush=True,
                )
                _write_batch_summary(args.output_dir, args.model, summary_rows)

    summary_rows.sort(key=lambda row: str(row["transcript_id"]).lower())
    _write_batch_summary(args.output_dir, args.model, summary_rows)
    if args.aggregate_csv is not None:
        _write_aggregate_csv(rows, args.output_dir, args.model, args.aggregate_csv)
    failures = [row for row in summary_rows if row["batch_status"] == "failed"]
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-prompt", type=str, default=DEFAULT_SMOKE_PROMPT)
    parser.add_argument("--smoke-max-tokens", type=int, default=200)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--skip-ids", type=str, default="")
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--transcript-id", type=str, default=None)
    parser.add_argument("--pick", choices=("first", "shortest"), default="first")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--masker-rounds", type=int, default=1)
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument("--refiller-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument(
        "--json-max-tokens",
        type=int,
        default=None,
        help="Defaults: qwen=81920, llama=4096, otherwise 6000.",
    )
    parser.add_argument(
        "--text-max-tokens",
        type=int,
        default=None,
        help="Defaults: qwen=81920, llama=4096, otherwise 6000.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Force OpenRouter chat-completions to send extra_body={'reasoning': {'enabled': False}}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_jsonl = args.input_jsonl.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.aggregate_csv is not None:
        args.aggregate_csv = args.aggregate_csv.resolve()
    model_default = _default_max_tokens_for_model(args.model)
    if args.json_max_tokens is None:
        args.json_max_tokens = model_default
    if args.text_max_tokens is None:
        args.text_max_tokens = model_default
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch and args.smoke_test:
        raise RuntimeError("--batch and --smoke-test cannot be used together")
    if args.batch and args.transcript_id:
        raise RuntimeError("--batch and --transcript-id cannot be used together")
    if args.smoke_test:
        return _run_smoke_test(args)
    if args.batch:
        return _run_batch(args)

    rows = _load_jsonl_rows(args.input_jsonl)
    row = _select_row(rows, args.transcript_id, args.pick)
    transcript_id = str(row.get("transcript_id", "")).strip()
    text = str(row.get("text", "")).strip()
    if not transcript_id or not text:
        raise RuntimeError("Selected row must contain transcript_id and text")

    prefix = _output_prefix(args.model, transcript_id)
    db_path = args.output_dir / f"{prefix}.db"
    input_jsonl = args.output_dir / f"{prefix}_input.jsonl"
    output_json = args.output_dir / f"{prefix}_output.json"
    output_csv = args.output_dir / f"{prefix}_output.csv"

    _reset_db(db_path)
    _write_jsonl_input(input_jsonl, transcript_id, text)
    _configure_openrouter(args, db_path)

    import db
    import phase0_init
    import pipeline

    print(f"[openrouter] model={args.model}")
    print(f"[openrouter] transcript_id={transcript_id} chars={len(text)}")
    print(f"[openrouter] db={db_path}")

    db.init_db()
    phase0_init.initialize_document(transcript_id, text)
    pipeline.run_one(transcript_id)

    result = _read_result(db_path, transcript_id)
    result.update(
        {
            "provider": "openrouter",
            "model": args.model,
            "source_jsonl": str(args.input_jsonl),
            "input_jsonl": str(input_jsonl),
        }
    )
    _write_outputs(output_csv, output_json, result)
    print(f"[openrouter] wrote {output_csv}")
    print(f"[openrouter] wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
