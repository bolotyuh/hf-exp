```
sudo yum install htop

conda create -n py310 python=3.10
conda init
source ~/.zshrc
```


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
