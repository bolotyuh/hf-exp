```
sudo yum install htop

conda create -n py310 python=3.10
conda init zsh
source ~/.zshrc
conda activate py310
```

git clone https://github.com/bolotyuh/hf-exp.git

https://docs.aws.amazon.com/fsx/latest/LustreGuide/exporting-files-hsm.html


## Attach file system
```
sudo amazon-linux-extras install -y lustre
sudo yum -y update kernel && sudo reboot
sudo amazon-linux-extras install -y lustre2.10

sudo mkdir /fsx
sudo mount -t lustre -o relatime,flock fs-05179a78ca1750276.fsx.us-east-1.amazonaws.com@tcp:/nqp5bbev /fsx
df -h
```

```
./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4

```