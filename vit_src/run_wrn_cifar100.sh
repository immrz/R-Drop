#!/usr/bin/env bash
pe=1
tbs=128
ebs=1024
ee=390
gas=1
mt="wrn_28_10"
ds="cifar100"
nstep=78125
at="23437 46875 62500"
ratio=0.2
lr=0.1
wd="5e-4"
dropout=0.3

data_dir="${AMLT_DATA_DIR:-data}"
output_dir="${AMLT_OUTPUT_DIR:-output}"
ngpu=${NPROC_PER_NODE:-1}

if [ ${ngpu} -gt 1 ]
then
    # if using multiple GPUs, do not do gradient accumulation; instead, adjust the per-GPU batch size
    let tbs=128/${ngpu}
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --aug_type cifar \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --decay_at ${at} \
        --decay_ratio ${ratio} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas ${gas} \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} \
        --prob_end ${pe} \
        --drop_rate ${dropout} $@
else
    python train.py \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --aug_type cifar \
        --model_type ${mt} \
        --dataset ${ds} \
        --num_steps ${nstep} \
        --decay_at ${at} \
        --decay_ratio ${ratio} \
        --learning_rate ${lr} \
        --weight_decay ${wd} \
        --train_batch_size ${tbs} \
        -gas ${gas} \
        --eval_batch_size ${ebs} \
        --eval_every ${ee} \
        --prob_end ${pe} \
        --drop_rate ${dropout} $@
fi
