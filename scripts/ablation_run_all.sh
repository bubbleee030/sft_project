#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -N 1
#SBATCH -J ablation_sft
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -p <partition>
#SBATCH --cpus-per-task=32
#SBATCH --output=ablation_sft_%j.out
#SBATCH --error=ablation_sft_%j.err

export LLM_TRAINING_PATH=${HOME}/llamafactory-workspace
export SIF_PATH=${HOME}/llamafactory.sif

DATASETS=("judge_sft_full" "judge_sft_no_tc_only" "judge_sft_no_simhash" "judge_sft_no_think" "judge_sft_raw")
BASE_MODEL="/Meta-Llama-3.1-8B-Instruct" 

for DATASET in "${DATASETS[@]}"; do
    echo "======================================"
    echo "Starting training for: $DATASET"
    echo "======================================"
    
    srun singularity exec --nv --home $LLM_TRAINING_PATH:/home/$USER $SIF_PATH bash -c "
      cd LLaMA-Factory && \
      FORCE_TORCHRUN=1 llamafactory-cli train \
          --stage sft \
          --do_train \
          --model_name_or_path $BASE_MODEL \
          --dataset $DATASET \
          --template llama3 \
          --finetuning_type lora \
          --output_dir saves/llama3-8b-judge/lora/$DATASET \
          --overwrite_cache \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --lr_scheduler_type cosine \
          --logging_steps 10 \
          --save_steps 500 \
          --learning_rate 5e-5 \
          --num_train_epochs 3.0 \
          --plot_loss \
          --fp16 \
          --report_to none
    "
done
