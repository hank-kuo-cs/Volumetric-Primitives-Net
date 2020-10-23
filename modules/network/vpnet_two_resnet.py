import torch
import torch.nn as nn
from torchvision.models import resnet18
from config import CUBOID_NUM, SPHERE_NUM, CONE_NUM, VP_CLAMP_MAX, VP_CLAMP_MIN, IS_DROPOUT, IS_SIGMOID, VOLUME_RESTRICT


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class VPNetTwoRes(nn.Module):
    """
    Volumetric Primitive Net:
    Assemble some volumetric primitives (spheres, cuboids and cones) to reconstruct target 3D mesh.
    This network use two resnet18 to predict the volume of theses volumetric primitives and their rotation and translation.
    """
    def __init__(self):
        super().__init__()
        self._vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

        self.volume_resnet = resnet18(pretrained=True)
        self.transform_resnet = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.volume_fc = self._make_linear(3 * self._vp_num)
        self.rotate_fc = self._make_linear(4 * self._vp_num)
        self.translate_fc = self._make_linear(3 * self._vp_num)

    def forward(self, imgs):
        volume_features = self.extract_feature(self.volume_resnet, imgs)
        transform_features = self.extract_feature(self.transform_resnet, imgs)

        volumes = self.volume_fc(volume_features)
        rotates = self.rotate_fc(transform_features)
        translates = self.translate_fc(transform_features)

        volumes, rotates, translates = self.restrict_range(volumes, rotates, translates)

        volumes = list(volumes.split(3, dim=1))
        rotates = list(rotates.split(4, dim=1))
        translates = list(translates.split(3, dim=1))

        volumes = self.restrict_volumes(volumes)

        return volumes, rotates, translates

    def extract_feature(self, model, imgs):
        out = model.conv1(imgs)
        out = model.bn1(out)
        out = model.relu(out)
        out = model.maxpool(out)

        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = model.layer4(out)

        out = self.avgpool(out)
        features = out.view(out.size(0), -1)

        return features

    def fix_volume_weight(self):
        for p in self.volume_resnet.parameters():
            p.requires_grad = False
        for p in self.volume_fc.parameters():
            p.requires_grad = False

    @staticmethod
    def restrict_range(volumes, rotates, translates):
        if IS_SIGMOID:
            volumes = sigmoid(volumes) + 0.1
            rotates = sigmoid(rotates)
            translates = tanh(translates)
        else:
            volumes = torch.clamp(volumes,  min=VP_CLAMP_MIN + 1e-8, max=VP_CLAMP_MAX)
            rotates = torch.clamp(rotates, min=-1, max=1)
            translates = torch.clamp(translates, min=-1, max=1)
        return volumes, rotates, translates

    @staticmethod
    def restrict_volumes(volumes):
        for i in range(len(volumes)):
            for j in range(3):
                volumes[i][:, j] = torch.div(volumes[i][:, j], VOLUME_RESTRICT[j])
        return volumes

    @staticmethod
    def _make_linear(output_dim: int):
        dropout_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, output_dim),
        )
        layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, output_dim),
        )
        return dropout_layers if IS_DROPOUT else layers
