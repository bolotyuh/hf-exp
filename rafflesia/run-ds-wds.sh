./distributed_train.sh 4 \
    --dataset wds/rafflesia \
    --data-dir /fsx/data \
    --train-split "{00000000..00002876}.tar" \
    --val-split "{00000000..00000059}.tar" \
    --model convnextv2_base.fcmae_ft_in22k_in1k \
    --pretrained \
    --experiment 'convnextv2_wds_initial' \
    --num-classes 23986 \
    --checkpoint-hist 2 \
    --amp \
    --torchcompile \
    --pin-mem \
    --batch-size 128 \
    --validation-batch-size 128 \
    --grad-accum-steps 1 \
    --workers 8 \
    --log-interval 100 \
    --epochs 5 \
    --sched cosine \
    --warmup-epochs 0 \
    --warmup-lr 1e-4 \
    --cooldown-epochs 0 \
    --decay-epochs 2 \
    --weight-decay 1e-8 \
    --opt adamw \
    --lr-base 2.5e-5 \
    --smoothing 0.1 \
    --log-wandb \
    --drop-path 0.2

# --lr 1e-4 \
# --lr-base 2.5e-5 \

# --save-images \
# --aa rand-m9-mstd0.5 \
# --log-wandb \
# --torchcompile \

