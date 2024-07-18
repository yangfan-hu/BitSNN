# BitSNN
This repo holds the codes for [BitSNN](https://doi.org/10.1109/TCDS.2024.3383428).

## Dependencies
* Python 3.8.8
* Pytorch 1.8.1
  
## Image Classification

### CIFAR-10

#### Training
python -u main.py --gpus 0 -a resnet20_hardtanh --data_path [DATA_PATH] --dataset cifar10 --epochs 600 --lr 0.1 -b 256 -bt 128 --lr_type cos --warm_up --weight_decay 5e-4 

### ImageNet

#### Training
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a resnet18_hardtanh -b 128 --workers 4 --lr 0.1 --lr_type cos --epochs 200 --warm_up --dali_cpu [DATA_PATH]





