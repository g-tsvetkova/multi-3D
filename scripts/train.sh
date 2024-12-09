export LLM_CONFIG='mesh-xl/mesh-xl-125m'
export NSAMPLE_PER_GPU=8
export SAMPLE_ROUNDS=100
export OUTPUT_DIR='./checkpoints'

python /root/MeshXL/main.py --model mesh_xl --dataset objaverse \
    --checkpoint_dir ./checkpoints  \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --sample_rounds $SAMPLE_ROUNDS \
    --dataset_num_workers 0 \