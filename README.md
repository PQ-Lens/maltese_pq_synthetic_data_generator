# Maltese PQ Synthetic Data Generator

Generate synthetic Maltese Parliamentary Question (PQ) and answer data in English from one Python entrypoint:

- `maltese_pq_synthetic_generator.py`

The legacy `pq_synth_core.py` module is now only a compatibility wrapper around that file.

## What It Generates

The generator creates fabricated but plausible records for six discourse categories:

| ID | Name | Type |
| --- | --- | --- |
| `A1` | Information-Seeking Questions | Question |
| `A2` | Assertion / Hidden-Accusation Questions | Question |
| `A3` | Request / Directive Questions | Question |
| `B1` | Replies (Direct Answers) | Answer |
| `B2` | Answers by Implication (Indirect/Partial) | Answer |
| `B3` | Non-Replies (Deferral/Referral/Reroute) | Answer |

Question categories use:

- `Date`
- `PQ No.`
- `MP`
- `Ministry (EN)`
- `Title (EN)`
- `Question (EN)`
- `Answer (EN)`

Answer categories use:

- `Date`
- `PQ No.`
- `MP`
- `Question (EN)`
- `Answer (EN)`

## Model Configuration

Model selection is dynamic and loaded from `.env`. Configure the models available on the server once, then pick one at runtime.

```env
# Local Ollama models available on this server
OLLAMA_MODELS=llama3.1:8b,qwen2.5:7b,mistral:7b
OLLAMA_BASE_URL=http://localhost:11434

# Gemini
GOOGLE_API_KEY=your_key_here
GEMINI_MODELS=gemini-2.5-flash,gemini-flash-latest

# Mistral
MISTRAL_API_KEY=your_key_here
MISTRAL_MODELS=mistral-small-latest,mistral-large-latest

# Optional defaults
LLM_PROVIDER=ollama
PQ_DEFAULT_N=2
PQ_DEFAULT_TEMPERATURE=0.7
PQ_DEFAULT_APPROACH=zero_shot
PQ_OUTPUT_DIR=pq_synthetic_outputs
PQ_MAX_OUTPUT_TOKENS=8192
PQ_API_RETRIES=3
PQ_API_RETRY_BACKOFF_SECONDS=2
PQ_CONTINUE_ON_ERROR=true
```

Single-model variables are still supported for backward compatibility:

- `OLLAMA_MODEL`
- `GEMINI_MODEL`
- `MISTRAL_MODEL`
- `LLM_MODEL`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas python-dotenv requests
```

For Ollama, make sure the server is running and the configured models exist:

```bash
curl http://localhost:11434/api/tags
ollama pull llama3.1:8b
```

## CLI Usage

List the model choices loaded from `.env`:

```bash
.venv/bin/python maltese_pq_synthetic_generator.py --list-models
```

Generate all six categories with the same per-category batch size:

```bash
.venv/bin/python maltese_pq_synthetic_generator.py \
  --model ollama:llama3.1:8b \
  --batch-size 10 \
  --api-retries 5
```

Generate selected categories only:

```bash
.venv/bin/python maltese_pq_synthetic_generator.py \
  --model gemini:gemini-2.5-flash \
  --batch-size 5 \
  --categories A1 A2 B1
```

If you run the script in an interactive terminal without `--model` or `--batch-size`, it presents the configured model list and prompts for the batch size. In non-interactive use, it falls back to `.env` defaults.

Transient provider failures are retried automatically. For example, HTTP `429`, `503`, retryable `5xx` responses, connection errors, and timeouts use exponential backoff controlled by `PQ_API_RETRIES` and `PQ_API_RETRY_BACKOFF_SECONDS`.

By default, a failed category writes a failure metrics CSV and the batch continues with the remaining categories. Use `--fail-fast` if you want the run to stop after the first category failure.

## Python API

```python
from maltese_pq_synthetic_generator import run_all_and_save, run_and_save

# One category
df, metrics = run_and_save(
    "A1",
    n=10,
    provider="ollama",
    model="llama3.1:8b",
)

# All categories, same batch size per category
results = run_all_and_save(
    n=10,
    provider="mistral",
    model="mistral-small-latest",
)
```

## Output

Generated CSV files are written to `pq_synthetic_outputs/` by default:

- data: `<CATEGORY>_<APPROACH>_<PROVIDER>_<MODEL>_<TIMESTAMP>.csv`
- metrics: `<CATEGORY>_<APPROACH>_<PROVIDER>_<MODEL>_<TIMESTAMP>_metrics.csv`
- combined run data: `all_categories_<APPROACH>_<PROVIDER>_<MODEL>_<TIMESTAMP>.csv`
- rolling summary: `summary_metrics.csv`

The combined run CSV contains all successful categories from that run and prepends:

- `Category ID`
- `Category Name`
- `Category Type`

## Validation

Each category is validated before saving:

- exact row count
- exact column names
- `Date` format as `DD/MM/YYYY`
- numeric `PQ No.`
- allowed MP names
- allowed ministry names when applicable

The generator retries once by default with corrective instructions if validation fails.
