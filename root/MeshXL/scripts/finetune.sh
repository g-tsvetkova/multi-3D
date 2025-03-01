export LLM_CONFIG=CH3COOK/mesh-xl-125m
export NSAMPLE_PER_GPU=1
export SAMPLE_ROUNDS=100
export OUTPUT_DIR='./output-finetune-mesh-xl'
# Find and kill all processes using the GPU
# Add memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision bf16 \
    main.py \
    --dataset shapenet_lamp \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --sample_rounds $SAMPLE_ROUNDS \
    --dataset_num_workers 0 \
    --max_epoch 1 \
    --finetune \