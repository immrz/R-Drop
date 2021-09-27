prt="True"
ps=1
pe=0.8
drp=0.2

# adjust batch size according to num of GPUs
ngpu=${NPROC_PER_NODE:-1}
let tbs=512/${ngpu}
ebs=600
ee=100
mt="efficientnet_b0"
ds="cifar100"
nstep=30000
ws=1500
lr=0.256
wd="1e-5"
opt="rmsprop"

echo "Using ${ngpu} GPUs"

if [ ${ngpu} -gt 1 ]
then
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
        --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
        --train_batch_size ${tbs} -gas 1 --eval_batch_size ${ebs} --eval_every ${ee} \
        --prob_start ${ps} --prob_end ${pe} --drop_rate ${drp} --fp16 --opt ${opt} $@
else
    python  train.py \
        --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
        --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
        --train_batch_size ${tbs} -gas 2 --eval_batch_size ${ebs} --eval_every ${ee} \
        --prob_start ${ps} --prob_end ${pe} --drop_rate ${drp} --fp16 --opt ${opt} $@
fi
