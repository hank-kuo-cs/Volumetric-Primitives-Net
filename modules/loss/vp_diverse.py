import torch
import torch.nn as nn
from .chamfer_distance import ChamferDistanceLoss
from config import L_VP_DIV, VP_NUM


class VPDiverseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cd_loss_func = ChamferDistanceLoss()

    def forward(self, translates: list, gt_points: torch.Tensor) -> torch.Tensor:
        self.check_parameters(translates)

        vp_center_points = torch.cat([t[:, None, :] for t in translates], 1)

        loss = self.cd_loss_func(vp_center_points, gt_points, w1=0.5, w2=1.0)
        return loss

    @staticmethod
    def check_parameters(translates):
        assert isinstance(translates, list)
        assert len(translates) == VP_NUM

