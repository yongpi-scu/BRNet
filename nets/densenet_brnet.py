import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['densenet169_brnet']


model_urls = {
    'densenet121': '/root/workspace/res/ckpt/pytorch/densenet/densenet121-a639ec97.pth',
    'densenet169': '/root/workspace/res/ckpt/pytorch/densenet/densenet169-b2777c0a.pth',
    'densenet201': '/root/workspace/res/ckpt/pytorch/densenet/densenet201-c1103571.pth',
    'densenet161': '/root/workspace/res/ckpt/pytorch/densenet/densenet161-8d451a50.pth',
}

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class MultiScale(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MultiScale, self).__init__()
        self.branch3x3dbl_1 = BasicConv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=1)
        self.branch3x3dbl_2 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        self.branch3x3dbl_3_1 = BasicConv2d(in_channels, out_channels*2, kernel_size=1, stride=1)
        self.branch3x3dbl_final = BasicConv2d(out_channels*7//2, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3dbl_1 = F.upsample_nearest(self.branch3x3dbl_1(F.avg_pool2d(x, kernel_size=3, stride=4)), size=x.shape[2:])
        branch3x3dbl_2 = F.upsample_nearest(self.branch3x3dbl_2(F.avg_pool2d(x, kernel_size=3, stride=2)), size=x.shape[2:])
        branch3x3dbl_3_1 = self.branch3x3dbl_3_1(x)
        x = torch.cat([branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3_1], 1)
        return self.branch3x3dbl_final(x)

def densenet169_brnet(pretrained=False, color_channels=3, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), color_channels=color_channels,
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        if "num_classes" in kwargs.keys() and kwargs["num_classes"] !=1000:
            del state_dict["classifier.weight"]
            del state_dict["classifier.bias"]
            del state_dict["features.conv0.weight"]
            del state_dict["features.norm0.weight"]
            del state_dict["features.norm0.bias"]
            del state_dict["features.norm0.running_mean"]
            del state_dict["features.norm0.running_var"]
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), color_channels=3,
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', MultiScale(color_channels, num_init_features)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=out.shape[2:], stride=1).view(features.size(0), -1)
        # out = F.max_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
