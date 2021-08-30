args=$@

prt="True"
ps=1
pe=0.8
drp=0.3
tbs=128  # total bs = 128 * 4 = 512
ebs=256
ee=100
gas=8  # mbs = 128 / 8 = 16
mt="efficientnetv2_m"
ds="cifar100"
nstep=10000
ws=250
lr=0.001
wd=0
alpha=1

python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
    --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
    --train_batch_size ${tbs} -gas ${gas} --eval_batch_size ${ebs} --eval_every ${ee} \
    --prob_start ${ps} --prob_end ${pe} --alpha ${alpha} --dropout_prob ${drp} --fp16 $@
