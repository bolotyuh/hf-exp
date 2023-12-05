python validate.py \
    --data-dir /home/gpubox3/projects/hf-exp/data \
    --split validation \
    --num-classes 23986 \
    --class-map /home/gpubox3/projects/hf-exp/model/classes-rafflesia.txt \
    --model convnextv2_base.fcmae_ft_in22k_in1k \
    --checkpoint /home/gpubox3/projects/hf-exp/model/checkpoint-0.pth.tar \
    --amp