#!/bin/bash

# Define models and their checkpoints using an associative array
declare -A models
models=(
    ["mesh_xl"]="/MeshXL/checkpoints/mesh_xl_checkpoint.pth"
    ["mesh_xl_mtp"]="/MeshXL/checkpoints/mesh_xl_mtp_checkpoint.pth"
    # Add more models here in the format ["ModelName"]="CheckpointPath"
)

# Output CSV file
output_csv="inference_times.csv"

# Write header to CSV
echo "Model,Iteration,Inference_Time_sec" > $output_csv

# Number of iterations per model
iterations=10

# Loop over each model
for model_name in "${!models[@]}"; do
    checkpoint_path=${models[$model_name]}
    echo "--------------------------------------------------"
    echo "Testing model: $model_name"
    echo "Checkpoint: $checkpoint_path"
    echo "--------------------------------------------------"

    for ((i=1; i<=iterations; i++)); do
        echo "Iteration $i for model $model_name"

        # Export environment variables
        export LLM_CONFIG='mesh-xl/mesh-xl-125m'  # Adjust if necessary
        export TEST_CKPT="$checkpoint_path"
        export MODEL="$model_name"  # This variable can be used inside main.py if needed

        # Run the main.py script using accelerate and capture the output
        # Assuming sample.sh runs main.py with the necessary arguments
        # If sample.sh is not set up, you can run the command directly here
        output=$(accelerate launch /root/MeshXL/main.py \
            --dataset objaverse \
            --n_max_triangles 800 \
            --n_discrete_size 128 \
            --llm "$LLM_CONFIG" \
            --model "$model_name" \
            --checkpoint_dir ./outputs \
            --dataset_num_workers 1 \
            --test_only)

        # Extract inference time using grep and awk
        # Assumes that "Inference Time: X.XXXX seconds" is printed
        inference_time=$(echo "$output" | grep "Inference Time" | awk '{print $3}')

        # Check if inference_time is not empty
        if [ -z "$inference_time" ]; then
            echo "Failed to capture inference time for model $model_name, iteration $i."
            inference_time="N/A"
        fi

        # Append the data to CSV
        echo "$model_name,$i,$inference_time" >> $output_csv

        # Optionally, print the captured inference time
        echo "Captured Inference Time: $inference_time seconds"
    done
done

echo "--------------------------------------------------"
echo "Inference times have been recorded in $output_csv"
echo "--------------------------------------------------"
