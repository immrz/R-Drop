#!/usr/bin/env bash

prd="checkpoint/ViT-B_16.npz"
tbs=512
ebs=64
ee=100
ims=224
gas=8
mt="ViT-B_16"
ds="imagenet"
nstep=20000
ws=500
lr=0.03
wd=0

data_dir="${AMLT_DATA_DIR:-/scratch/ILSVRC2012}"
output_dir="${AMLT_OUTPUT_DIR:-output}"
ngpu=${NPROC_PER_NODE:-1}

if [ ${ngpu} -gt 1 ]
then
    let tbs=512/${ngpu}
    let gas=8/${ngpu}
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --pretrained_dir ${prd} \
        --img_size ${ims} \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --warmup_steps ${ws} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas ${gas} \
        --fp16 \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} $@
else
    python train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --pretrained_dir ${prd} \
        --img_size ${ims} \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --warmup_steps ${ws} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas ${gas} \
        --fp16 \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} $@
fi
