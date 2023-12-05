./distributed_train.sh 8 \
    --dataset wds/rafflesia \
    --data-dir /fsx/data \
    --train-split "{00000000..00002876}.tar" \
    --val-split "{00000000..00000059}.tar" \
    --model convnextv2_base.fcmae_ft_in22k_in1k \
    --pretrained \
    --experiment 'convnextv2_wds_aug' \
    --num-classes 23986 \
    --checkpoint-hist 2 \
    --amp \
    --aa rand-m9-mstd0.5 \
    --torchcompile \
    --pin-mem \
    --lr-base 2.5e-4 \
    --batch-size 128 \
    --validation-batch-size 512 \
    --grad-accum-steps 1 \
    --workers 8 \
    --log-interval 100 \
    --epochs 30 \
    --sched cosine \
    --sched-on-updates \
    --warmup-epochs 0 \
    --cooldown-epochs 0 \
    --weight-decay 1e-8 \
    --opt adamw \
    --smoothing 0.1 \
    --log-wandb \
    --drop-path 0.1


# --amp-dtype bfloat16 \
# --lr-base 2.5e-4 \
# --decay-epochs 2 \
# --warmup-lr 1e-4 \
# --lr 1e-4 \
# --lr-base 2.5e-5 \
# --save-images \
# --aa rand-m9-mstd0.5 \
# --log-wandb \
# --torchcompile \

