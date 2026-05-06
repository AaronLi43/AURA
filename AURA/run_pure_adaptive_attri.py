#!/usr/bin/env python3
"""Run the AURA pipeline with ONLY dynamically generated privacy attributes.

Thin wrapper over `run_expanded_privacy.py` that:
  - Forces `--no-base-attributes` (no base 8).
  - Defaults `--export-dir` to `output/pure_adaptive_attri/` next to this script.
  - Defaults `--name-prefix` to `pure_adaptive_attri`.
  - Emits a rewritten CSV at `<export-dir>/<name-prefix>_rewritten.csv`.

Any additional CLI flags accepted by `run_expanded_privacy.py` (e.g. `--reset-db`,
`--direct-intent-workers`, `--attribute-workers`, `--feedback-rounds`,
`--reid-threshold`, `--direct-intent-model`, `--attribute-model`,
`--max-new-attributes`, `--max-total-attributes`, `--output-csv`) are forwarded as-is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pipeline_config as cfg
import run_expanded_privacy


DEFAULT_EXPORT_DIR = cfg.BASE_DIR / "output" / "pure_adaptive_attri"
DEFAULT_NAME_PREFIX = "pure_adaptive_attri"


def _arg_provided(args: list[str], name: str) -> bool:
    for token in args:
        if token == name or token.startswith(f"{name}="):
            return True
    return False


def _inject_defaults(user_args: list[str]) -> list[str]:
    injected: list[str] = list(user_args)

    if not _arg_provided(injected, "--export-dir"):
        injected.extend(["--export-dir", str(DEFAULT_EXPORT_DIR)])

    if not _arg_provided(injected, "--name-prefix"):
        injected.extend(["--name-prefix", DEFAULT_NAME_PREFIX])

    if not _arg_provided(injected, "--no-base-attributes"):
        injected.append("--no-base-attributes")

    if not _arg_provided(injected, "--max-new-attributes"):
        injected.extend(["--max-new-attributes", "4"])

    if not _arg_provided(injected, "--max-total-attributes"):
        injected.extend(["--max-total-attributes", "12"])

    return injected


def main() -> int:
    user_args = sys.argv[1:]
    sys.argv = [sys.argv[0], *_inject_defaults(user_args)]
    print(f"[pure-adaptive] forwarding args: {sys.argv[1:]}")
    return run_expanded_privacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
