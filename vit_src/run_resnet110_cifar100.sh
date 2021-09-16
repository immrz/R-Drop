#!/usr/bin/env bash

prt="False"
ps=1
pe=0.5
tbs=1024
ebs=1024
ee=50
gas=2
mt="resnet110"
ds="cifar100"
nstep=25000
ws=250
lr=0.8
wd="1e-4"
alpha=1

data_dir="${AMLT_DATA_DIR:-data}"
output_dir="${AMLT_OUTPUT_DIR:-output}"
ngpu=${NPROC_PER_NODE:-1}

if [ ${ngpu} -gt 1 ]
then
    # if using multiple GPUs, do not do gradient accumulation; instead, adjust the per-GPU batch size
    let tbs=1024/${ngpu}
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --pretrained ${prt} \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --warmup_steps ${ws} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas 1 \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} \
        --prob_start ${ps} \
        --prob_end ${pe} \
        --alpha ${alpha} $@
else
    python train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --pretrained ${prt} \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --warmup_steps ${ws} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas ${gas} \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} \
        --prob_start ${ps} \
        --prob_end ${pe} \
        --alpha ${alpha} $@
fi
