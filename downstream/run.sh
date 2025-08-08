#!/bin/bash

# Arrays for different configurations

# Batch sizes
config_batch_size=(8 16 32 64)
# Learning rates
config_learning_rate=(3e-4 1e-4 1e-5)
# Model configurations and checkpoints in the format "config_file ckpt_file" (space-separated)
config_model=(
    "./downstream/model_configs/XE/transformer_3mer_50M_CLM.json ./downstream/gpt_clm/last.ckpt"
    "./downstream/model_configs/XE/transformer_3mer_50M_MLM.json ./downstream/gpt_mlm/last.ckpt"
)
# Dataset configurations path
config_dataset=(./downstream/data/MRFP/config.json)

# Dataset seeds to run for each task
config_dataset_seed=(0 1 2)

# Fine-tuning methods
config_finetune_method=(none)

# Output path
output_path=(./downstream/mrfp.json)

# GPU IDs to use for running jobs (0-indexed)
gpu_ids=(0)

declare -A running_jobs

# Function to run a single job
run_job() {
    local c1=$1
    local c2=$2
    local c3="$3"
    local c4=$4
    local c5=$5
    local c6=$6
    local c7=$7

    if [ ${#gpu_ids[@]} -eq 0 ]; then
        return 1
    fi

    local gpu=${gpu_ids[0]}
    gpu_ids=("${gpu_ids[@]:1}")

    # Split 'c3' into config_file and ckpt_file
    IFS=" " read -r config_file ckpt_file <<< "$c3"

    echo "Running job with config: $c1 $c2 $config_file $ckpt_file $c4 $c5 on GPU $gpu"

    python run_downstream.py --gpu-id $gpu --batch-size $c1 --learning-rate-head $c2 --model-config "$config_file" --model-path "$ckpt_file" --dataset-config $c4 --output-path $c5 --seed-index $c6 --finetune $c7 &
    local pid=$!
    running_jobs[$pid]=$gpu
    echo "Started job with PID $pid on GPU $gpu"
}

cleanup_jobs() {
    for pid in "${!running_jobs[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "Job $pid completed. Freeing GPU ${running_jobs[$pid]}"
            gpu_ids+=(${running_jobs[$pid]})
            unset running_jobs[$pid]
        fi
    done
}

# Counter for GPU assignment
gpu_available() {
    [ ${#gpu_ids[@]} -gt 0 ]
}

# Nested loops for configurations
for c7 in "${config_finetune_method[@]}"; do
    for c1 in "${config_batch_size[@]}"; do
        for c2 in "${config_learning_rate[@]}"; do
            for c3 in "${config_model[@]}"; do
                for c6 in "${config_dataset_seed[@]}"; do
                    for c4 in "${config_dataset[@]}"; do
                        for c5 in "${output_path[@]}"; do
                            while ! run_job $c1 $c2 "$c3" $c4 $c5 $c6 $c7; do
                                sleep 5
                                cleanup_jobs
                            done
                        done
                    done
                done
            done
        done
    done
done

# Wait for any remaining jobs to finish
wait

echo "All jobs completed"
