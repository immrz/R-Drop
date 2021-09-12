#!/usr/bin/env bash

python train.py \
    --model resnet110 \
    --dataset cifar100 \
    --output_dir outputs \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 256 \
    --learning_rate 0.8 \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --max_steps 10000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --logging_steps 10 \
    --save_strategy epoch \
    --fp16 False \
    --dataloader_num_workers 8 \
    --disable_tqdm True \
    --report_to tensorboard $@
