import torch
from .inception_v3_brnet import Inception_V3_BRNet
from .inceptionresnetv2_brnet import InceptionResNetV2BRNet

model_urls = {
    'Inception_V3_BRNet': '/root/workspace/res/ckpt/pytorch/inception/inception_v3_google-1a9a5a14.pth',
    'InceptionResNetV2BRNet': '/root/workspace/res/ckpt/pytorch/inception/inceptionresnetv2-520b38e4.pth',
}

def load_pretrain(net, model_name, exclusions):
    """Restore model from checkpoint file."""
    pretrained_dict = torch.load(model_urls[model_name])
    for key in list(pretrained_dict.keys()):
        for exclusion in exclusions:
            if exclusion in key:
                pretrained_dict.pop(key)
                break
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    print("%s model restored"%model_name)
    net.load_state_dict(model_dict)
    return net

def inception_v3_brnet(pretrained=True, num_classes=2, drop_rate=0.5, color_channels=3):
    net = Inception_V3_BRNet(num_classes=num_classes, drop_rate=drop_rate, color_channels=color_channels)
    exclusions = ['fc.weight', 'fc.bias', 'AuxLogits', 'Conv2d_1a_3x3']
    net = load_pretrain(net, 'Inception_V3_BRNet', exclusions)
    return net
    
def inception_resnet_v2_brnet(pretrained=True, num_classes=2, drop_rate=0.5, color_channels=3):
    net = InceptionResNetV2BRNet(num_classes=num_classes, drop_rate=drop_rate, color_channels=color_channels)
    exclusions = ['last_linear','conv2d_1a']
    net = load_pretrain(net, 'InceptionResNetV2BRNet', exclusions)
    return net