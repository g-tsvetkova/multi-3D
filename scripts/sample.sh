export LLM_CONFIG='mesh-xl/mesh-xl-125m'
# the checkpoint mush align with the $LLM_CONFIG
export TEST_CKPT='/MeshXL/checkpoints/checkpoint.pth'

accelerate launch main.py \
    --dataset objaverse \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm mesh-xl/mesh-xl-125m \
    --model mesh_xl_mtp \
    --checkpoint_dir ./outputs \
    --batchsize_per_gpu 1 \
    --test_ckpt $TEST_CKPT \
    --sample_rounds 100 \
    --dataset_num_workers 0 \
    --test_only
   