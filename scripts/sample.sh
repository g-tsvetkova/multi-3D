export LLM_CONFIG='mesh-xl/mesh-xl-125m'
# the checkpoint mush align with the $LLM_CONFIG
export TEST_CKPT='/root/MeshXL/checkpoints'
# export MODEL = model


accelerate launch /root/MeshXL/main.py \
    --dataset shapenet_lamp \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --llm $LLM_CONFIG \
    --model mesh_xl_mtp \
    --checkpoint_dir $TEST_CKPT \
    --dataset_num_workers 1 \
    --sample_rounds 1 \
    --test_only