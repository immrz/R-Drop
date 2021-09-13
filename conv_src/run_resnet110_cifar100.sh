#!/usr/bin/env bash

python train.py \
    --model resnet110 \
    --dataset cifar100 \
    --output_dir output \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 1024 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 0.8 \
    --weight_decay 1e-4 \
    --max_grad_norm 1 \
    --max_steps 25000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --save_strategy epoch \
    --fp16 True \
    --dataloader_num_workers 8 \
    --disable_tqdm True \
    --report_to tensorboard \
    --load_best_model_at_end True $@
