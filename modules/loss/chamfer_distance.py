import torch
import torch.nn as nn
from config import CD_W1, CD_W2, BATCH_SIZE


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist_min1, _ = torch.min(dist1, dim=2)
        dist_min2, _ = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean()
        loss2 = dist_min2.mean()

        loss = (CD_W1 * loss1 + CD_W2 * loss2) / BATCH_SIZE
        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(0) == BATCH_SIZE
        assert points.size(-1) == 3
