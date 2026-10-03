#!/bin/bash
#SBATCH --job-name=eval_benchmarks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Load environment
# source activate venv

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: sbatch eval_benchmarks.sh <path_to_model>"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"

# Run lm-evaluation-harness
lm_eval --model vllm \
  --model_args pretrained=${MODEL_PATH},dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096,enforce_eager=True \
  --tasks tmmluplus,cmmlu,hellaswag,xwinograd_zh \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./benchmark_results/

# Note for DRCD and FGC: You will need to implement custom task configurations 
# in lm-evaluation-harness to run them, as they are not built-in standard tasks.
