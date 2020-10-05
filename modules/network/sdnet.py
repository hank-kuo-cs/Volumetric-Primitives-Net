import torch.nn as nn
from torchvision.models import resnet18


class SDNet(nn.Module):
    """
    Sphere Deformation Net (just for experiment):
    Deform single sphere to reconstruct target 3D meshes.
    This network is to predict the offset of all vertices composing the sphere.
    """
    def __init__(self):
        super().__init__()
        self._model = resnet18(pretrained=True)
        self._model.avgpool = self._make_avg_pool()
        self._model.deform = self._make_linear(386 * 3)

    def forward(self, imgs):
        features = self.extract_feature(imgs)
        vertices_offset = self._model.deform(features)
        return vertices_offset

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
    def _make_linear(output_dim: int):
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
