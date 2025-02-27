export LLM_CONFIG='mesh-xl/mesh-xl-125m'
export BATCHSIZE_PER_GPU=2
export CUDA_LAUNCH_BLOCKING=1
export CHKPT_DIR = /root/MeshXL/scripts/checkpoints

accelerate launch \
    --config_file ./config/deepspeed_stage2.yaml \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision bf16 \
    main.py \
    --dataset shapenet_lamp \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --warm_lr_iters -1 \
    --base_lr 1e-6 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $CHKPT_DIR \
    --batchsize_per_gpu $BATCHSIZE_PER_GPU \
    --dataset_num_workers 0 \
    --augment \
    --eval_every_iteration 10000 \
    --save_every 20000 \
    --max_epoch 1024 \
    --train_from_scratch