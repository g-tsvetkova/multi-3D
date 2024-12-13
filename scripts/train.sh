export LLM_CONFIG='mesh-xl/mesh-xl-125m'
export NSAMPLE_PER_GPU=1
export SAMPLE_ROUNDS=100
export OUTPUT_DIR='./checkpoints'

# python /root/MeshXL/main.py --model mesh_xl --dataset objaverse \
#     --checkpoint_dir ./checkpoints  \
#     --n_max_triangles 800 \
#     --n_discrete_size 128 \
#     --llm $LLM_CONFIG \
#     --model mesh_xl \
#     --checkpoint_dir $OUTPUT_DIR \
#     --batchsize_per_gpu $NSAMPLE_PER_GPU \
#     --max_epoch 100 \
#     --sample_rounds $SAMPLE_ROUNDS \
#     --dataset_num_workers 0 \

accelerate launch main.py \
    --dataset objaverse \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --dataset_num_workers 1 \
    --max_epoch 100
    