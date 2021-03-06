import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from ..render import VertexRenderer
from config import SILHOUETTE_LOSS_FUNC


class SilhouetteLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = L1Loss() if SILHOUETTE_LOSS_FUNC == 'L1' else MSELoss()

    def forward(self, predict_meshes: list, gt_silhouettes: torch.Tensor,
                dists: torch.Tensor, elevs: torch.Tensor, azims: torch.Tensor) -> torch.Tensor:
        predict_silhouettes = []
        for i in range(len(predict_meshes)):
            _, predict_silhouette, _ = VertexRenderer.render(predict_meshes[i], dists[i], elevs[i], azims[i])
            predict_silhouettes.append(predict_silhouette.permute(0, 3, 1, 2))

        predict_silhouettes = torch.cat(predict_silhouettes)

        loss = self.loss_func(predict_silhouettes, gt_silhouettes)
        return loss
