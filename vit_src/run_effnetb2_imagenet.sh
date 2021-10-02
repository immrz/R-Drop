data_dir="${AMLT_DATA_DIR:-/scratch/ILSVRC2012}"
output_dir="${AMLT_OUTPUT_DIR:-imagenet_output}"
ngpu=${NPROC_PER_NODE:-1}

python -m torch.distributed.launch --nproc_per_node=${ngpu} timm_train.py \
    --model efficientnet_b2 \
    -b 32 \
    --sched step \
    --epochs 450 \
    --decay-epochs 2.4 \
    --decay-rate .97 \
    --opt rmsproptf \
    --opt-eps .001 \
    -j 8 \
    --warmup-lr 1e-6 \
    --weight-decay 1e-5 \
    --drop 0.3 \
    --drop-connect 0.2 \
    --model-ema \
    --model-ema-decay 0.9999 \
    --aa rand-m9-mstd0.5 \
    --remode pixel \
    --reprob 0.2 \
    --amp \
    --lr .016 \
    --train-split trainset \
    --val-split val \
    --data_dir ${data_dir} \
    --output ${output_dir} $@
