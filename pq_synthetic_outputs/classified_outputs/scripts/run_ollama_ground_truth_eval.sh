#!/usr/bin/env bash
set -euo pipefail

cd /home/davis/maltese_pq_eval/pq_synthetic_outputs

rm -f \
  ground_truth_synthetic_data_ollama_gemma2_2b_* \
  ground_truth_synthetic_data_ollama_gemma3_4b_*

export PYTHONUNBUFFERED=1

python3 ollama_eval.py \
  ground_truth_synthetic_data.csv \
  --model gemma2:2b \
  --batch-size 5 \
  --max-tokens 1024 \
  --api-retries 2 \
  --retry-backoff 2 \
  --output-prefix ground_truth_synthetic_data_ollama_gemma2_2b

python3 ollama_eval.py \
  ground_truth_synthetic_data.csv \
  --model gemma3:4b \
  --batch-size 5 \
  --max-tokens 1024 \
  --api-retries 2 \
  --retry-backoff 2 \
  --output-prefix ground_truth_synthetic_data_ollama_gemma3_4b
