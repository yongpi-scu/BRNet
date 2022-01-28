# BRNet
code for "Automated assessment of BI-RADS categories for ultrasound images using multi-scale neural networks with an order-constrained loss function"

## Requirements
- PyTorch 1.0+
- Python 3.x
## Running
train:
```bash
python train.py --config cfgs/brnet.yaml --net brnet --model mi_model --gpu 0 --seed 666
```
eval:
```bash
python eval.py --config cfgs/brnet.yaml --net brnet --model mi_model --gpu 0 --save_name BRNet_seed_666 --ckpt_path path-to-checkpoint.pth
```
## Data
The [data](data/) folder contains a portion of our dataset.