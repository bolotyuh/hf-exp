python train_timm2.py \
    --dataset wds/rafflesia \
    --data-dir /mnt/disk1/wds-rafflesia-subdataset/train \
    --train-split "ds-train-{00000..00009}.tar" \
    --val-split "ds-train-{00000..00009}.tar" \
    --model convnextv2_base.fcmae_ft_in22k_in1k \
    --pretrained \
    --experiment 'convnextv2_wds' \
    --num-classes 20 \
    --checkpoint-hist 2 \
    --amp \
    --pin-mem \
    --batch-size 64 \
    --validation-batch-size 64 \
    --grad-accum-steps 1 \
    --workers 4 \
    --log-interval 1 \
    --epochs 10 \
    --weight-decay 0. \
    --opt adamw \
    --lr-base 2.5e-5 \
    --smoothing 0.1 \
    --sched cosine \
    --warmup-epochs 0 \
    --min-lr 5e-5 \
    --warmup-lr 1e-4 \
    --cooldown-epochs 0 \
    --drop-path 0.1

# --save-images \
# --aa rand-m9-mstd0.5 \
# --log-wandb \
# --torchcompile \

