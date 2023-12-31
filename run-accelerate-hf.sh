export WANDB_PROJECT=rafflesia-accelerate


accelerate launch --config_file ./default_config.yaml train_hf.py \
    --model_name_or_path microsoft/swin-base-patch4-window7-224-in22k \
    --train_dir /fsx/data/val \
    --validation_dir /fsx/data/val \
    --output_dir ./rafflesia-outputs-swin2-local/ \
    --remove_unused_columns False \
    --do_train \
    --fp16 \
    --label_smoothing_factor 0.1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine" \
    --logging_strategy steps \
    --logging_steps 100 \
    --num_train_epochs 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --auto_find_batch_size True \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --load_best_model_at_end True \
    --report_to wandb \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --seed 41

# --do_eval \
