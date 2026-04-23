# Maltese PQ Synthetic Data Generator

Generate synthetic Maltese Parliamentary Question (PQ) and answer data in English using either Google Gemini or a local Ollama model.

This project produces fabricated but plausible records for six discourse categories:
- Question categories: `A1`, `A2`, `A3`
- Answer categories: `B1`, `B2`, `B3`

The core implementation is now available in:
- `maltese_pq_synthetic_generator.py`

The original notebook is still included in:
- `maltese_pq_synthetic_generator_gemini_api.ipynb`

Generated CSV files are written to:
- `pq_synthetic_outputs/`

## What It Generates

For each selected category, the notebook generates `n` synthetic rows (default `10`) and validates structure before saving.

### Category Definitions

| ID | Name | Type | Description |
| --- | --- | --- | --- |
| `A1` | Information-Seeking Questions | Question | Neutral, specific requests for facts (counts, dates, locations, status). |
| `A2` | Assertion / Hidden-Accusation Questions | Question | Challenge-style questions implying criticism or accountability concerns. |
| `A3` | Request / Directive Questions | Question | Requests to table documents, provide lists, confirm contracts, publish breakdowns, etc. |
| `B1` | Replies (Direct Answers) | Answer | Direct responses with requested facts or clear yes/no with details. |
| `B2` | Answers by Implication (Indirect/Partial) | Answer | Partial or indirect responses with updates, conditions, or vague timelines. |
| `B3` | Non-Replies (Deferral/Referral/Reroute) | Answer | Deferrals, references to other PQs, or rerouting to other ministers. |

### Output Schemas

Question categories (`A1–A3`) use:
- `Date`
- `PQ No.`
- `MP`
- `Ministry (EN)`
- `Title (EN)`
- `Question (EN)`
- `Answer (EN)`

Answer categories (`B1–B3`) use:
- `Date`
- `PQ No.`
- `MP`
- `Question (EN)`
- `Answer (EN)`

## How It Works

The generation pipeline:
1. Loads environment variables for provider, model, and output settings.
2. Builds category-specific prompts and calls either Gemini or Ollama.
3. Forces JSON output and parses model text safely.
4. Validates:
   - exact row count (`n`)
   - exact column names
   - `Date` format as `DD/MM/YYYY`
   - numeric `PQ No.`
5. Retries once with corrective instructions if validation fails.
6. Saves valid output to CSV in `pq_synthetic_outputs/`.

## Data Controls and Constraints

Prompt constraints enforce:
- formal parliamentary English tone
- fictional content only (no verbatim real records)
- allowed MP names from a fixed list
- allowed ministries from a fixed list when ministry column applies
- realistic PQ number ranges (e.g., `10000–40000`)

## Prerequisites

- Python 3.12+ (notebook metadata currently uses Python `3.12.3`)
- Jupyter environment (or VS Code notebook support)
- Google AI Studio API key for Gemini, or a running Ollama instance for local models

Required environment variable for Gemini:
- `GOOGLE_API_KEY`

Optional environment variables:
- `LLM_PROVIDER` (defaults to `ollama`)
- `LLM_MODEL` (defaults to `llama3.1:8b`)
- `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`)
- `PQ_DEFAULT_N` (defaults to `2` in code for lightweight testing)
- `PQ_DEFAULT_TEMPERATURE`
- `PQ_DEFAULT_APPROACH`
- `PQ_OUTPUT_DIR`
- `PQ_MAX_OUTPUT_TOKENS`

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install pandas python-dotenv requests
```

3. Add environment variables in your shell or `.env` file.

Gemini example:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here
LLM_MODEL=gemini-2.5-flash
```

Ollama example:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

Use the Python module directly, or import it into the notebook:

```python
from maltese_pq_synthetic_generator import run_and_save, set_model, set_provider

set_provider("gemini")
set_model("gemini-2.5-flash")
df, metrics = run_and_save("A1", n=10, approach="one_shot")
```

If you prefer notebooks, you can still open and run `maltese_pq_synthetic_generator_gemini_api.ipynb` cell by cell.

Main helper:
- `run_and_save(category_id: str, n: int = 10, temperature: float = 0.7, approach: str | None = None)`

Example calls (one category at a time):
- `run_and_save("A1")`
- `run_and_save("A2")`
- `run_and_save("A3")`
- `run_and_save("B1")`
- `run_and_save("B2")`
- `run_and_save("B3")`

Optional batch loop is provided in the notebook to generate all six categories sequentially.

## Output Files

Each run writes one CSV with a timestamped filename:
- `pq_synthetic_outputs/<CATEGORY_ID>_<YYYYMMDD_HHMMSS>.csv`

Example:
- `pq_synthetic_outputs/A1_20260226_171530.csv`

## Troubleshooting

- If token or length issues occur:
  - reduce `n`
  - reduce prompt verbosity
  - lower `maxOutputTokens` in `call_gemini_chat_json()`
- If API key errors occur:
  - verify key is correct
  - check Google AI Studio key restrictions/referrer settings
- If validation fails repeatedly:
  - rerun the cell
  - lower temperature
  - generate fewer rows

## Project Structure

```text
.
├── README.md
├── maltese_pq_synthetic_generator.py
├── maltese_pq_synthetic_generator_gemini_api.ipynb
└── pq_synthetic_outputs/
```

## Important Notes

- This tool is for synthetic data generation only.
- Generated text should be reviewed before research or production use.
- Do not treat generated rows as factual parliamentary records.



## Steps to run locally 

1. Create a virtual environment:
python3 -m venv .venv
source .venv/bin/activate


2. Install dependencies:
pip install pandas python-dotenv requests


3. Make sure Ollama is running and the model exists:
curl http://localhost:11434/api/tags
ollama pull llama3.1:8b


4. Set environment variables, either in a .env file:

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
PQ_DEFAULT_N=2


5. Run like this:
python -c 'from maltese_pq_synthetic_generator import run_and_save; run_and_save("A1", n=2)'




