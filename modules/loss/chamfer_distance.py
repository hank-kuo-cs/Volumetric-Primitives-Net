import torch
import torch.nn as nn


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1: float = 1.0, w2: float = 1.0) -> torch.Tensor:
        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist_min1, _ = torch.min(dist1, dim=2)
        dist_min2, _ = torch.min(dist2, dim=2)

        return (w1 * torch.sum(dist_min1) + w2 * torch.sum(dist_min2)) / points1.size(0)
