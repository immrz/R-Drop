#!/usr/bin/env bash

output_dir="${AMLT_OUTPUT_DIR:-output}"
ngpu=${NPROC_PER_NODE:-1}

if [ -z ${AMLT_OUTPUT_DIR+x} ]
then
    # on local machine
    nworker=4
else
    # on ITP server
    nworker=32
fi

echo "Using ${ngpu} GPUs."

if [ ${ngpu} -gt 1 ]
then
    let tbs=1024/${ngpu}
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --model resnet110 \
        --dataset cifar100 \
        --output_dir ${output_dir} \
        --evaluation_strategy epoch \
        --per_device_train_batch_size ${tbs} \
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
        --dataloader_num_workers ${nworker} \
        --disable_tqdm True \
        --report_to tensorboard \
        --load_best_model_at_end True $@
else
    python train.py \
        --model resnet110 \
        --dataset cifar100 \
        --output_dir ${output_dir} \
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
        --dataloader_num_workers ${nworker} \
        --disable_tqdm True \
        --report_to tensorboard \
        --load_best_model_at_end True $@
fi
