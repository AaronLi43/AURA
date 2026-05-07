# CODE

Reference implementation of the **AURA** privacy-rewriting pipeline used in
our NeurIPS 2026 submission.  Given a JSONL of conversation transcripts,
AURA produces an anonymized rewrite with controllable privacy scope.

| Entry point | Purpose |
|---|---|
| [`pipeline.py`](pipeline.py) | The four-phase AURA pipeline (mask → refill → attack → keep) over a single transcript, orchestrated through a SQLite scratch DB. |
| [`run_expanded_privacy.py`](run_expanded_privacy.py) | Adaptive-privacy variant: probes each transcript with web search to discover dynamic privacy attributes, then runs `pipeline.py` on top of the expanded scope. |
| [`run_pure_adaptive_attri.py`](run_pure_adaptive_attri.py) | Thin wrapper around `run_expanded_privacy.py` that disables the eight base attributes (`--no-base-attributes`) so only the dynamically discovered ones are protected. |
| [`run_qwen_expanded_batch.py`](run_qwen_expanded_batch.py) | Batch driver that runs the adaptive pipeline over a directory of transcripts, optionally swapping the LLM provider to OpenRouter (e.g. for Qwen variants). |
| [`run_openrouter_sample.py`](run_openrouter_sample.py) | Single-transcript driver against OpenRouter, useful for sanity-checking on-device models without touching the rest of the pipeline. |

The four phase modules ([`phase0_init.py`](phase0_init.py),
[`phase1_masker.py`](phase1_masker.py),
[`phase1_refiller.py`](phase1_refiller.py),
[`phase2_attacker.py`](phase2_attacker.py),
[`phase2_keeper.py`](phase2_keeper.py)) and the SQLite layer
[`db.py`](db.py) are imported by the entry points above; they are not
designed to be invoked directly.

## Setup

```bash
cd AURA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
$EDITOR .env   # add OPENAI_API_KEY (and OPENROUTER_API_KEY if you'll use Qwen)
```

## Quick start

The repo ships a tiny synthetic input
([`input/example_transcripts.jsonl`](input/example_transcripts.jsonl)) so
the end-to-end pipeline is runnable out of the box.  Each row must have
`conversation_id` and `user_message` keys; the pipeline rewrites
`user_message` and writes the anonymized output to a CSV alongside the
intermediate per-transcript SQLite scratch DB.

```bash
# Phase 0–2 over every transcript in the example JSONL.
python pipeline.py --reset-db

# Adaptive-privacy variant (Phase 0 + dynamic attribute discovery + Phase 1–2).
python run_expanded_privacy.py --reset-db

# Adaptive variant with ONLY the dynamically discovered scope.
python run_pure_adaptive_attri.py --reset-db
```

Outputs are written under `output/`:

```
output/
├── <name-prefix>_rewritten.csv               # the anonymized transcripts
└── adaptive_attri/
    ├── original_direct_intent.json           # web-search re-id on originals
    └── expanded_privacy_attributes.json      # discovered per-transcript scopes
```

## Running on your own data

`pipeline.py` and the adaptive wrappers read JSONL with these schemas:

```jsonc
// input/example_transcripts.jsonl  (one object per line)
{
  "conversation_id": "S00000001",
  "user_message": "Assistant: ... User: ..."
}
```

Pass `--input-jsonl path/to/yours.jsonl` to override the default.  All
parameters (worker counts, retry budget, masker/refiller models, etc.) are
exposed via `--help`.  The CLI accepts these top-level options on every
entry point above:

| Flag | Default | Notes |
|---|---|---|
| `--input-jsonl` | `input/example_transcripts.jsonl` | Source transcripts. |
| `--reset-db` | off | Wipe SQLite scratch state before the run. |
| `--feedback-rounds` | 5 | Phase-1 mask/refill iterations per transcript. |
| `--variations` | 4 | Refill candidates explored per round. |
| `--reid-threshold` | (model default) | Severity at which Phase-2 forces another mask round. |
| `--max-new-attributes`, `--max-total-attributes` | 4 / 12 | Caps on dynamically discovered attributes (adaptive only). |

### Switching the masker / refiller / attacker LLM

`pipeline_config.py` reads the model identifiers from environment
variables, defaulting to `gpt-4.1`.  To run the on-device Qwen variants
used in the paper, install [OpenRouter](https://openrouter.ai/) credits and:

```bash
export NB_LLM_PROVIDER=openrouter
export NB_MASKER_MODEL=qwen/qwen3.5-27b
export NB_REFILLER_MODEL=qwen/qwen3.5-27b
python run_qwen_expanded_batch.py --reset-db
```

## Outputs

Each transcript ends up with a `<conversation_id>.db` SQLite scratch file
plus a row in the consolidated `<name-prefix>_rewritten.csv`.  The DB
captures every prompt/response in every phase and is useful for debugging.
Both files are listed in [`../.gitignore`](../.gitignore) so they are not
accidentally committed.

## Privacy

This release deliberately omits:

* The original Anthropic re-id-shared transcript JSONL (replaced by the
  synthetic `example_transcripts.jsonl`).
* The web-search re-id JSON for the originals (`output/adaptive_attri/`),
  because each candidate description is verbatim text from the original
  transcripts.
* All SQLite scratch DBs from the runs reported in the paper.

If you reproduce the paper numbers on a private workspace, the resulting
`*.db`, `output/<run>_rewritten.csv`, and
`output/adaptive_attri/*_original_direct_intent.json` files should be
treated as sensitive — they may contain or be keyed by content that is not
fit for public release.

See [`../EVAL/README.md`](../EVAL/README.md)
for the matching re-identification and utility evaluation harness.
