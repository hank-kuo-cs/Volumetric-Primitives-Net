import torch
import torch.nn as nn
from torchvision.models import resnet18
from config import CUBOID_NUM, SPHERE_NUM, CONE_NUM, VP_CLAMP_MAX, VP_CLAMP_MIN

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class VPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

        self._model = resnet18(pretrained=True)
        self._model.avgpool = self._make_avg_pool()

        self._model.volume = self._make_linear(3 * self._vp_num)
        self._model.rotate = self._make_linear(4 * self._vp_num)
        self._model.translate = self._make_linear(3 * self._vp_num)

    def forward(self, x):
        out = self._model.conv1(x)
        out = self._model.bn1(out)
        out = self._model.relu(out)
        out = self._model.maxpool(out)

        out = self._model.layer1(out)
        out = self._model.layer2(out)
        out = self._model.layer3(out)
        out = self._model.layer4(out)

        out = self._model.avgpool(out)
        features = out.view(out.size(0), -1)

        volumes = torch.clamp(self._model.volume(features), min=VP_CLAMP_MIN, max=VP_CLAMP_MAX)
        rotates = torch.clamp(self._model.rotate(features), min=-1, max=1)
        translates = torch.clamp(self._model.translate(features), min=-1, max=1)

        # volumes = sigmoid(self._model.volume(features))
        # rotates = tanh(self._model.rotate(features))
        # translates = tanh(self._model.translate(features))

        volumes = volumes.split(3, dim=1)
        rotates = rotates.split(4, dim=1)
        translates = translates.split(3, dim=1)

        # volumetric_primitives = []
        #
        # for i in range(self._vp_num):
        #     vp = torch.cat([volumes[i], rotates[i], translates[i]], dim=1)
        #     volumetric_primitives.append(vp)

        return volumes, rotates, translates

    @staticmethod
    def _make_linear(output_dim: int):
        # cuboid: w, h, d, rotate(4), translate(3) -> 10
        # sphere: r(3), rotate(4), translate(3) -> 10
        # cone: r(2), h, rotate(4), translate(3) -> 10
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, output_dim),
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
