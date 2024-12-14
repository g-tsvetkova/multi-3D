export LLM_CONFIG='mesh-xl/mesh-xl-125m'
# the checkpoint mush align with the $LLM_CONFIG
export TEST_CKPT='/MeshXL/checkpoints/checkpoint.pth'


accelerate launch /root/MeshXL/main.py \
    --dataset objaverse \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl_mtp \
    --checkpoint_dir ./outputs \
    --dataset_num_workers 1 \
    --test_only