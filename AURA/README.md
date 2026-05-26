# CODE

Reference implementation of the **AURA** privacy-rewriting pipeline used in
our NeurIPS 2026 submission.  Given a JSONL of conversation transcripts,
AURA produces an anonymized rewrite with controllable privacy scope.

| Entry point | Purpose |
|---|---|
| [`pipeline.py`](pipeline.py) | The four-phase AURA pipeline (mask → refill → attack → keep) over a single transcript, orchestrated through a SQLite scratch DB. |
| [`run_expanded_privacy.py`](run_expanded_privacy.py) | Adaptive-privacy variant: probes each transcript with web search to discover dynamic privacy attributes, then runs `pipeline.py` on top of the expanded scope. |
| [`run_pure_adaptive_attri.py`](run_pure_adaptive_attri.py) | Thin wrapper around `run_expanded_privacy.py` that disables the eight base attributes (`--no-base-attributes`) so only the dynamically discovered ones are protected. |
| [`run_openrouter_sample.py`](run_openrouter_sample.py) | Single-transcript driver against OpenRouter, useful for sanity-checking on-device models without touching the rest of the pipeline. |

The four phase modules ([`phase0_init.py`](phase0_init.py),
[`phase1_masker.py`](phase1_masker.py),
[`phase1_refiller.py`](phase1_refiller.py),
[`phase2_attacker.py`](phase2_attacker.py),
[`phase2_keeper.py`](phase2_keeper.py)) and the SQLite layer
[`db.py`](db.py) are imported by the entry points above; they are not
designed to be invoked directly.

## Setup

Run these from the repo root (the folder that contains both `AURA/` and
`EVAL/`):

```bash
cd aura   # your clone directory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp AURA/.env.example AURA/.env
nano AURA/.env   # add OPENAI_API_KEY (and OPENROUTER_API_KEY if you'll use Qwen)
```

Then `cd AURA` before running any of the commands below.  On macOS, use
`python3` if `python` is not installed.

## Quick start

The repo ships a tiny synthetic input
([`input/example_transcripts.jsonl`](input/example_transcripts.jsonl)) so
the end-to-end pipeline is runnable out of the box.  Each row must have
`conversation_id` and `user_message` keys; the pipeline rewrites
`user_message` and writes the anonymized output to a CSV alongside the
intermediate per-transcript SQLite scratch DB.

```bash
cd AURA   # the pipeline folder (this directory)
```

### Privacy scope × provider

| Scope | What is protected | GPT (OpenAI) | OpenRouter (Qwen) |
|---|---|---|---|
| **Base only** | 8 predefined attributes only | `python run_expanded_privacy.py --reset-db --only-base-attri` | Set OpenRouter vars in `.env` (below), then the same command |
| **Adaptive** | Base 8 + dynamically discovered attributes | `python run_expanded_privacy.py --reset-db` | Set OpenRouter vars in `.env`, then the same command |
| **Pure adaptive** | Dynamically discovered attributes only | `python run_pure_adaptive_attri.py --reset-db` | Set OpenRouter vars in `.env`, then the same command |

Default CSV outputs:

| Scope | Output CSV |
|---|---|
| Base only | `output/adaptive_attri/nobranch_rewritten.csv` |
| Adaptive | `output/adaptive_attri/nobranch_rewritten.csv` |
| Pure adaptive | `output/pure_adaptive_attri/pure_adaptive_attri_rewritten.csv` |

**Recommended first run:**

```bash
python run_expanded_privacy.py --reset-db
# → output/adaptive_attri/nobranch_rewritten.csv
```

**OpenRouter** (all scopes; set in `.env`, re-id still uses OpenAI):

```bash
# Add to .env (see .env.example), then:
python run_expanded_privacy.py --reset-db --only-base-attri   # base 8 on Qwen
# python run_expanded_privacy.py --reset-db                  # adaptive on Qwen
# python run_pure_adaptive_attri.py --reset-db               # pure adaptive on Qwen
```

**Basic 4-phase pipeline** (base 8 only, no CSV export; SQLite in `pipeline.db`):

```bash
python phase0_init.py --reset-db
python pipeline.py run-all
```

### Common CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--reset-db` | off | Delete the SQLite scratch DB before running. |
| `--only-base-attri` | off | Skip adaptive attribute discovery; protect only the 8 base attributes. |
| `--no-base-attributes` | off | Pure-adaptive mode (forced by `run_pure_adaptive_attri.py`). |
| `--skip-reid` | off | Rewrite only; skip web-search re-id on rewritten outputs. |
| `--input` | `input/example_transcripts.jsonl` | Source transcript JSONL. |
| `--export-dir` | scope-specific | Output directory for CSV, attribute JSON, and re-id artifacts. |
| `--name-prefix` | `nobranch` / `pure_adaptive_attri` | Filename prefix for exported artifacts. |
| `--ids` | all rows | Comma-separated transcript IDs to process. |
| `--feedback-rounds` | `1` | Re-run on still-re-identified transcripts (skipped with `--only-base-attri`). |
| `--max-new-attributes` | `12` | Cap on new dynamic attributes per transcript per round. |
| `--max-total-attributes` | `12` | Cap on total attributes per transcript. |
| `--direct-intent-model` | `gpt-5.1` | Web-search re-id model (OpenAI only). |
| `--attribute-model` | `gpt-4.1` | Dynamic attribute generation model. |

OpenRouter model selection (set in `.env`):

| Variable | Example | Notes |
|---|---|---|
| `NB_LLM_PROVIDER` | `openrouter` | Route pipeline LLM calls through OpenRouter. |
| `NB_MASKER_MODEL` | `qwen/qwen3.5-27b` | Applied to masker/refiller/attacker/keeper/init unless overridden. |
| `NB_DISABLE_REASONING` | `1` | Recommended for Qwen on OpenRouter. |

Outputs are written under `output/`:

```
output/
├── <name-prefix>_rewritten.csv               # the anonymized transcripts
└── adaptive_attri/
    ├── original_direct_intent.json           # web-search re-id on originals
    └── expanded_privacy_attributes.json      # discovered per-transcript scopes
```

## Running on your own data

The adaptive wrappers and `phase0_init.py` read JSONL with this schema:

```jsonc
// input/example_transcripts.jsonl  (one object per line)
{
  "conversation_id": "S00000001",
  "user_message": "Assistant: ... User: ..."
}
```

Pass `--input path/to/yours.jsonl` to `run_expanded_privacy.py` or
`phase0_init.py` to override the default.  Model IDs, worker counts, and
retry budgets are configured in [`pipeline_config.py`](pipeline_config.py)
and via environment variables; run each script with `--help` for its flags.

| Entry point | Notable CLI flags |
|---|---|
| [`run_expanded_privacy.py`](run_expanded_privacy.py) | `--reset-db`, `--only-base-attri`, `--input`, `--export-dir`, `--name-prefix`, `--skip-reid`, `--feedback-rounds`, `--max-new-attributes`, `--max-total-attributes` |
| [`run_pure_adaptive_attri.py`](run_pure_adaptive_attri.py) | Forwards all `run_expanded_privacy.py` flags; forces `--no-base-attributes` |
| [`phase0_init.py`](phase0_init.py) | `--reset-db`, `--input`, `--ids` |
| [`pipeline.py`](pipeline.py) | `run-one --doc-id …`, `run-all [--max-iter N] [--max-workers N]` |

### Switching the masker / refiller / attacker LLM

`pipeline_config.py` reads model identifiers from environment variables,
defaulting to `gpt-4.1`.  To run Qwen via OpenRouter, add these to `.env`
(see [`.env.example`](.env.example)) and use `run_expanded_privacy.py` or
`run_pure_adaptive_attri.py`:

```bash
NB_LLM_PROVIDER=openrouter
NB_MASKER_MODEL=qwen/qwen3.5-27b
NB_REFILLER_MODEL=qwen/qwen3.5-27b
NB_ATTACKER_MODEL=qwen/qwen3.5-27b
NB_KEEPER_MODEL=qwen/qwen3.5-27b
NB_INIT_MODEL=qwen/qwen3.5-27b
NB_DISABLE_REASONING=1

python run_expanded_privacy.py --reset-db --only-base-attri
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
