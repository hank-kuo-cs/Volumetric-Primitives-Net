import torch
import torch.nn as nn
from torchvision.models import resnet18
from config import CUBOID_NUM, SPHERE_NUM, CONE_NUM, VP_CLAMP_MAX, VP_CLAMP_MIN, IS_DROPOUT, IS_SIGMOID, VOLUME_RESTRICT


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class VPNet(nn.Module):
    """
    Volumetric Primitive Net:
    Assemble some volumetric primitives (spheres, cuboids and cones) to reconstruct target 3D mesh.
    This network is to predict the volume of theses volumetric primitives and their rotation and translation.
    """
    def __init__(self):
        super().__init__()
        self._vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

        self._model = resnet18(pretrained=True)
        self._model.avgpool = self._make_avg_pool()

        self._model.volume = self._make_linear(3 * self._vp_num)
        self._model.rotate = self._make_linear(4 * self._vp_num)
        self._model.translate = self._make_linear(3 * self._vp_num)

    def forward(self, imgs):
        features = self.extract_feature(imgs)

        volumes = self._model.volume(features)
        rotates = self._model.rotate(features)
        translates = self._model.translate(features)

        volumes, rotates, translates = self.restrict_range(volumes, rotates, translates)

        volumes = volumes.split(3, dim=1)
        rotates = rotates.split(4, dim=1)
        translates = translates.split(3, dim=1)

        volumes = self.restrict_volumes(volumes)

        return volumes, rotates, translates

    def extract_feature(self, imgs):
        out = self._model.conv1(imgs)
        out = self._model.bn1(out)
        out = self._model.relu(out)
        out = self._model.maxpool(out)

        out = self._model.layer1(out)
        out = self._model.layer2(out)
        out = self._model.layer3(out)
        out = self._model.layer4(out)

        out = self._model.avgpool(out)
        features = out.view(out.size(0), -1)

        return features

    @staticmethod
    def restrict_range(volumes, rotates, translates):
        if IS_SIGMOID:
            volumes = sigmoid(volumes)
            rotates = tanh(rotates)
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

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
