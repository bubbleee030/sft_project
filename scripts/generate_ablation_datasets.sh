#!/bin/bash
# generate_ablation_datasets.sh
# Automates the generation of 5 datasets for the ablation study

set -e

# Navigate to project root (assuming script is in scripts/)
cd "$(dirname "$0")/.."

# Base output directory
BASE_OUT="./data_ablation"
mkdir -p "$BASE_OUT"

echo "=========================================="
echo "Generating Ablation Datasets..."
echo "=========================================="

# 1. Full Pipeline (SFT v2)
echo "1. Generating Full Pipeline (SFT v2) dataset..."
python cli.py process-training-data \
    --step1_input_dir ./eval_results \
    --step1_output_dir $BASE_OUT/step1_full \
    --step2_output_dir $BASE_OUT/step2_full \
    --step3_output_dir $BASE_OUT/datasets \
    --step3_output_filename judge_sft_full.jsonl

# 2. No TC-Only (Keep Simplified)
echo "2. Generating No TC-Only dataset (keeping simplified)..."
python3 cli.py process-training-data \
    --step1_input_dir ./eval_results \
    --step1_output_dir $BASE_OUT/step1_no_tc \
    --step2_output_dir $BASE_OUT/step2_no_tc \
    --step3_output_dir $BASE_OUT/datasets \
    --step3_output_filename judge_sft_no_tc_only.jsonl \
    --keep-simplified

# 3. No SimHash
echo "3. Generating No SimHash dataset..."
python cli.py process-training-data \
    --step1_input_dir ./eval_results \
    --step1_output_dir $BASE_OUT/step1_no_simhash \
    --step2_output_dir $BASE_OUT/step2_no_simhash \
    --step3_output_dir $BASE_OUT/datasets \
    --step3_output_filename judge_sft_no_simhash.jsonl \
    --no-simhash

# 4. No Think Filter
echo "4. Generating No Think Filter dataset..."
python cli.py process-training-data \
    --step1_input_dir ./eval_results \
    --step1_output_dir $BASE_OUT/step1_no_think \
    --step2_output_dir $BASE_OUT/step2_no_think \
    --step3_output_dir $BASE_OUT/datasets \
    --step3_output_filename judge_sft_no_think.jsonl \
    --keep-think

# 5. Raw Baseline (All filters off)
echo "5. Generating Raw Baseline dataset..."
python cli.py process-training-data \
    --step1_input_dir ./eval_results \
    --step1_output_dir $BASE_OUT/step1_raw \
    --step2_output_dir $BASE_OUT/step2_raw \
    --step3_output_dir $BASE_OUT/datasets \
    --step3_output_filename judge_sft_raw.jsonl \
    --keep-simplified \
    --no-simhash \
    --keep-think

echo "=========================================="
echo "All datasets generated in $BASE_OUT/datasets/"
echo "=========================================="
