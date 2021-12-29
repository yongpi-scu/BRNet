# BRNet
code for "Automated assessment of BI-RADS categories for ultrasound images using multi-scale neural networks with an order-constrained loss function"

## Requirements
- PyTorch 1.0+
- Python 3.x
## Running
train:
```bash
python train.py --config cfgs/brnet.yaml --net inception_v3_brnet --model mixup_model --gpu 0
```
eval:
```bash
python eval.py --config cfgs/brnet.yaml --net inception_v3_brnet --model mixup_model --gpu 0 --save_name InceptionV3-BRNet_trail_1 --ckpt_path path-to-checkpoint.pth
```
## Data
The [data](data/) folder contains a portion of our dataset.