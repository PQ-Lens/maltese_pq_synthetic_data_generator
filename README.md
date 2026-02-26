# Maltese PQ Synthetic Data Generator

Generate synthetic Maltese Parliamentary Question (PQ) and answer data in English using the Google Gemini API.

This project produces fabricated but plausible records for six discourse categories:
- Question categories: `A1`, `A2`, `A3`
- Answer categories: `B1`, `B2`, `B3`

The core implementation is in:
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

The notebook pipeline:
1. Loads environment variables (`GOOGLE_API_KEY`, optional `GEMINI_MODEL`).
2. Builds category-specific prompts and calls Gemini `generateContent`.
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
- Google AI Studio API key

Required environment variable:
- `GOOGLE_API_KEY`

Optional environment variable:
- `GEMINI_MODEL` (defaults to `gemini-flash-latest`)

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies used by the notebook:

```bash
pip install pandas python-dotenv requests
```

3. Add environment variables in your shell or `.env` file:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-latest
```

## Usage

Open and run `maltese_pq_synthetic_generator_gemini_api.ipynb` cell by cell.

Main helper:
- `run_and_save(category_id: str, n: int = 10, temperature: float = 0.7)`

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
├── maltese_pq_synthetic_generator_gemini_api.ipynb
└── pq_synthetic_outputs/
```

## Important Notes

- This tool is for synthetic data generation only.
- Generated text should be reviewed before research or production use.
- Do not treat generated rows as factual parliamentary records.
