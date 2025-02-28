export LLM_CONFIG='mesh-xl/mesh-xl-125m'
export NSAMPLE_PER_GPU=1
export OUTPUT_DIR='./checkpoints'
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch \
    --mixed_precision bf16 \
    main.py \
    --dataset shapenet_lamp \
    --n_max_triangles 800 \
    --n_discrete_size 128 \
    --warm_lr_iters -1 \
    --base_lr 1e-6 \
    --llm $LLM_CONFIG \
    --model mesh_xl \
    --checkpoint_dir $OUTPUT_DIR \
    --batchsize_per_gpu $NSAMPLE_PER_GPU \
    --dataset_num_workers 1 \
       --eval_every_iteration 10000 \
    --save_every 20000 \
    --max_epoch 1024 \
   #  --train_from_scratch
    