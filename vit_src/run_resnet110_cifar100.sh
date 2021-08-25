args=$@

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

python train.py --pretrained ${prt} --model_type ${mt} --dataset ${ds} \
    --num_steps ${nstep} --warmup_steps ${ws} --learning_rate ${lr} --weight_decay ${wd} \
    --train_batch_size ${tbs} -gas ${gas} --eval_batch_size ${ebs} --eval_every ${ee} \
    --prob_start ${ps} --prob_end ${pe} --alpha ${alpha} $@
