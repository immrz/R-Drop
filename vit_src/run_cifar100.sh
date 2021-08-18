args=$@

prt="False"
ps=0.8
pe=0.8
tbs=128
ebs=1024
ee=400
gas=1
mt="resnet110"
ds="cifar100"
nstep=110000
ws=1000
lr=0.1
wd="1e-4"
alpha=1

python train.py --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
    --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
    --train_batch_size ${tbs} -gas ${gas} --eval_batch_size ${ebs} --eval_every ${ee} \
    --prob_start ${ps} --prob_end ${pe} --alpha ${alpha} $@
