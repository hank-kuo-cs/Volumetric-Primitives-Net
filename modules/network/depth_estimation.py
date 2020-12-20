"""
Depth estimation network follow GenRe (https://github.com/xiumingzhang/GenRe-ShapeHD)
to implement a U-ResNet structure
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from .reverse_resnet import reverse_u_resnet18


class DepthEstimationNet(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, input_planes=3):
        super().__init__()

        # Encoder
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        module_list.append(
            nn.Sequential(
                resnet.conv1 if input_planes == 3 else in_conv,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        )
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)

        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None

        # Decoder
        module_list = list()
        reverse_resnet = reverse_u_resnet18(out_planes=1)
        module_list.append(reverse_resnet.layer1)
        module_list.append(reverse_resnet.layer2)
        module_list.append(reverse_resnet.layer3)
        module_list.append(reverse_resnet.layer4)
        module_list.append(
            nn.Sequential(
                reverse_resnet.deconv1,
                reverse_resnet.bn1,
                reverse_resnet.relu,
                reverse_resnet.deconv2
            )
        )
        self.decoder = nn.ModuleList(module_list)

    def forward(self, x):
        # Encode
        feat = x
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]

        # Decode
        x = feat_maps[-1]
        for idx, f in enumerate(self.decoder):
            x = f(x)
            if idx < len(self.decoder) - 1:
                feat_map = feat_maps[-(idx + 2)]
                assert feat_map.shape[2:4] == x.shape[2:4]
                x = torch.cat((x, feat_map), dim=1)

        return x
