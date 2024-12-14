#!/bin/bash

# Define models and their checkpoints
declare -A models
models=(
    ["MeshXL"]="/root/MeshXL/checkpoints/checkpoint.pth"
    ["MeshXL_MTP"]="/root/MeshXL/checkpoints/checkpoint.pth"
    # Add more models here if needed
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
        export TEST_CKPT=$checkpoint_path

        # Run the sample.sh script and capture the output
        # Assuming sample.sh runs main.py and prints "Inference Time: X.XXXX seconds"
        output=$(bash sample.sh)

        # Extract inference time using grep and awk
        inference_time=$(echo "$output" | grep "Inference Time" | awk '{print $3}')

        # Append to CSV
        echo "$model_name,$i,$inference_time" >> $output_csv

        # Optionally, print the captured inference time
        echo "Captured Inference Time: $inference_time seconds"
    done
done

echo "--------------------------------------------------"
echo "Inference times have been recorded in $output_csv"
echo "--------------------------------------------------"
