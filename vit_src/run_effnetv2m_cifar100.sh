args=$@

prt="True"
ps=1
pe=0.8
drp=0.3

# adjust batch size according to num of GPUs
ngpu=${NPROC_PER_NODE:-1}
let tbs=512/${ngpu}
let gas=${tbs}/16
ebs=128
ee=100
mt="efficientnetv2_m"
ds="cifar100"
nstep=10000
ws=100
lr=0.01
wd=0
alpha=1

echo "Using ${ngpu} GPUs"

if [ ${ngpu} -gt 1 ]
then
    python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
        --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
        --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
        --train_batch_size ${tbs} -gas ${gas} --eval_batch_size ${ebs} --eval_every ${ee} \
        --prob_start ${ps} --prob_end ${pe} --alpha ${alpha} --dropout_prob ${drp} --fp16 $@
else
    python  train.py \
        --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
        --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
        --train_batch_size ${tbs} -gas ${gas} --eval_batch_size ${ebs} --eval_every ${ee} \
        --prob_start ${ps} --prob_end ${pe} --alpha ${alpha} --dropout_prob ${drp} --fp16 $@
fi
