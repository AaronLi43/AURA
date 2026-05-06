#!/usr/bin/env python3
"""Run all NIPS samples in parallel with expanded privacy scope."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from run_openrouter_sample import (
    DEFAULT_INPUT_JSONL,
    DEFAULT_MODEL,
    _configure_openrouter,
    _default_max_tokens_for_model,
    _load_jsonl_rows,
    _read_result,
    _reset_db,
    _select_row,
    _slug,
    _write_jsonl_input,
    _write_outputs,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_ATTRIBUTE_JSON = (
    BASE_DIR / "output" / "adaptive_attri" / "expanded_privacy_attributes.json"
)
DEFAULT_MAX_PARALLEL = 27
DEFAULT_MASKER_ROUNDS = 5
DEFAULT_VARIATIONS = 4


def _coerce_attribute_scope(raw_payload: dict, transcript_id: str):
    from run_expanded_privacy import (
        _coerce_privacy_attribute,
        _merge_attribute_lists,
    )

    raw_attributes = raw_payload.get("attributes", [])
    if not isinstance(raw_attributes, list):
        raw_attributes = []

    base_count = int(raw_payload.get("base_count", 0) or 0)
    base_attrs = [
        attr
        for raw_attr in raw_attributes[:base_count]
        if (attr := _coerce_privacy_attribute(raw_attr)) is not None
    ]

    per_doc = raw_payload.get("per_transcript_new_attributes", {})
    if not isinstance(per_doc, dict):
        per_doc = {}
    extras_raw = per_doc.get(transcript_id)
    if extras_raw is None:
        lower_to_key = {str(key).lower(): key for key in per_doc}
        extras_raw = per_doc.get(lower_to_key.get(transcript_id.lower(), ""))
    if not isinstance(extras_raw, list):
        extras_raw = []

    extra_attrs = [
        attr
        for raw_attr in extras_raw
        if (attr := _coerce_privacy_attribute(raw_attr)) is not None
    ]
    return _merge_attribute_lists(base_attrs, extra_attrs), len(base_attrs), len(extra_attrs)


def _load_attribute_scope(attribute_json: Path, transcript_id: str):
    payload = json.loads(attribute_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expanded attribute file must contain a JSON object: {attribute_json}"
        )
    return _coerce_attribute_scope(payload, transcript_id)


def _expanded_output_prefix(model: str, transcript_id: str) -> str:
    return f"expanded_{_slug(model)}_{_slug(transcript_id)}"


def _expanded_sample_output_paths(
    output_dir: Path,
    model: str,
    transcript_id: str,
) -> tuple[Path, Path]:
    prefix = _expanded_output_prefix(model, transcript_id)
    return output_dir / f"{prefix}_output.csv", output_dir / f"{prefix}_output.json"


def _read_sample_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _run_expanded_sample(args: argparse.Namespace) -> int:
    rows = _load_jsonl_rows(args.input_jsonl)
    row = _select_row(rows, args.worker_transcript_id, "first")
    transcript_id = str(row.get("transcript_id", "")).strip()
    text = str(row.get("text", "")).strip()
    if not transcript_id or not text:
        raise RuntimeError("Selected row must contain transcript_id and text")

    prefix = _expanded_output_prefix(args.model, transcript_id)
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
    from run_expanded_privacy import apply_attribute_scope

    scope, base_count, extra_count = _load_attribute_scope(
        args.attribute_json,
        transcript_id,
    )
    apply_attribute_scope(scope)

    print(f"[openrouter-expanded] model={args.model}")
    print(
        f"[openrouter-expanded] transcript_id={transcript_id} "
        f"chars={len(text)} attrs={len(scope)} "
        f"(base={base_count} + extra={extra_count})"
    )
    print(f"[openrouter-expanded] db={db_path}")

    db.init_db()
    phase0_init.initialize_document(transcript_id, text)
    pipeline.run_one(transcript_id)

    result = _read_result(db_path, transcript_id)
    result.update(
        {
            "provider": "openrouter",
            "model": args.model,
            "privacy_scope": "expanded",
            "attribute_json": str(args.attribute_json),
            "attribute_count": len(scope),
            "base_attribute_count": base_count,
            "expanded_attribute_count": extra_count,
            "source_jsonl": str(args.input_jsonl),
            "input_jsonl": str(input_jsonl),
        }
    )
    _write_outputs(output_csv, output_json, result)
    print(f"[openrouter-expanded] wrote {output_csv}")
    print(f"[openrouter-expanded] wrote {output_json}")
    return 0


def _write_expanded_batch_summary(
    output_dir: Path,
    model: str,
    rows: list[dict],
) -> None:
    summary_json = output_dir / "expanded_batch_summary.json"
    summary_csv = output_dir / "expanded_batch_summary.csv"
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
                "elapsed_s",
                "log_path",
            ]
        )
        for row in rows:
            sample_csv, output_json = _expanded_sample_output_paths(
                output_dir,
                model,
                row["transcript_id"],
            )
            sample = _read_sample_csv(sample_csv)
            writer.writerow(
                [
                    row["transcript_id"],
                    row["batch_status"],
                    row.get("returncode", ""),
                    output_json.exists(),
                    sample.get("status", ""),
                    sample.get("final_privacy_score", ""),
                    sample.get("final_text", ""),
                    row.get("elapsed_s", ""),
                    row.get("log_path", ""),
                ]
            )
    print(f"[expanded-batch] wrote {summary_json}")
    print(f"[expanded-batch] wrote {summary_csv}")


def _build_expanded_worker_command(
    args: argparse.Namespace,
    transcript_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-transcript-id",
        transcript_id,
        "--input-jsonl",
        str(args.input_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--attribute-json",
        str(args.attribute_json),
        "--model",
        args.model,
        "--max-parallel",
        str(args.max_parallel),
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
    ]
    if args.disable_reasoning:
        cmd.append("--disable-reasoning")
    if not args.skip_completed:
        cmd.append("--no-skip-completed")
    return cmd


def _run_expanded_batch_worker(
    args: argparse.Namespace,
    transcript_id: str,
    log_dir: Path,
) -> dict:
    start = time.monotonic()
    log_path = log_dir / f"{_slug(transcript_id)}.log"
    cmd = _build_expanded_worker_command(args, transcript_id)
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
    _, output_json = _expanded_sample_output_paths(
        args.output_dir,
        args.model,
        transcript_id,
    )
    return {
        "transcript_id": transcript_id,
        "batch_status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "output_json_exists": output_json.exists(),
        "elapsed_s": elapsed_s,
        "log_path": str(log_path),
    }


def _run_expanded_batch(args: argparse.Namespace) -> int:
    rows = _load_jsonl_rows(args.input_jsonl)
    target_ids: list[str] = []
    summary_rows: list[dict] = []

    for row in rows:
        transcript_id = str(row.get("transcript_id", "")).strip()
        if not transcript_id:
            continue
        _, output_json = _expanded_sample_output_paths(
            args.output_dir,
            args.model,
            transcript_id,
        )
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

    log_dir = args.output_dir / "expanded_batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[expanded-batch] total_rows={len(rows)} pending={len(target_ids)} "
        f"max_parallel={args.max_parallel} attributes={args.attribute_json}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {
            executor.submit(
                _run_expanded_batch_worker,
                args,
                transcript_id,
                log_dir,
            ): transcript_id
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
                        f"[expanded-batch] progress completed={completed_count}/{len(target_ids)} "
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
                    log_path.write_text(
                        f"Expanded batch worker failed: {type(exc).__name__}: {exc}\n",
                        encoding="utf-8",
                    )
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
                    f"[expanded-batch] {result['batch_status']} {transcript_id} "
                    f"returncode={result['returncode']} elapsed_s={result['elapsed_s']} "
                    f"progress={completed_count}/{len(target_ids)}",
                    flush=True,
                )
                _write_expanded_batch_summary(args.output_dir, args.model, summary_rows)

    summary_rows.sort(key=lambda row: str(row["transcript_id"]).lower())
    _write_expanded_batch_summary(args.output_dir, args.model, summary_rows)
    failures = [row for row in summary_rows if row["batch_status"] == "failed"]
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attribute-json", type=Path, default=DEFAULT_ATTRIBUTE_JSON)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--masker-rounds", type=int, default=DEFAULT_MASKER_ROUNDS)
    parser.add_argument("--variations", type=int, default=DEFAULT_VARIATIONS)
    parser.add_argument("--refiller-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--json-max-tokens", type=int, default=None)
    parser.add_argument("--text-max-tokens", type=int, default=None)
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--worker-transcript-id",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_jsonl = args.input_jsonl.resolve()
    args.output_dir = args.output_dir.resolve()
    args.attribute_json = args.attribute_json.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_default = _default_max_tokens_for_model(args.model)
    if args.json_max_tokens is None:
        args.json_max_tokens = model_default
    if args.text_max_tokens is None:
        args.text_max_tokens = model_default

    if args.worker_transcript_id:
        return _run_expanded_sample(args)
    return _run_expanded_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
