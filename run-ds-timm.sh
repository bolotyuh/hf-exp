./distributed_train.sh 4 --data-dir /fsx/data \
    --train-split val \
    --val-split val \
    --model convnextv2_base --pretrained \
    --project-name 'Mushroom-large' \
    --experiment 'd0-convnextv2-fcmae_ft_in22k_in1k' \
    --num-classes 23986 \
    --aa rand-m9-mstd0.5 \
    --checkpoint-hist 2 \
    --log-interval 1 \
    --amp \
    --pin-mem \
    --batch-size 64 \
    --validation-batch-size 128 \
    --grad-accum-steps 1 \
    --workers 4 \
    --epochs 1 \
    --weight-decay 1e-8 \
    --opt adamw \
    --lr 1e-4 \
    --smoothing 0.1 \
    --sched cosine \
    --warmup-epochs 0 \
    --min-lr 5e-5 \
    --warmup-lr 1e-4 \
    --cooldown-epochs 0 \
    --drop-path 0.1

# --torchcompile \
