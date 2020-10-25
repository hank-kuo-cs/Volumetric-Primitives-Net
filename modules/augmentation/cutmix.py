import torch
from config import DEVICE


def cut_mix_data(rgbs: torch.Tensor, silhouettes: torch.Tensor, view_center_points: torch.Tensor):
    check_parameters(rgbs, silhouettes, view_center_points)
    B, C, H, W = rgbs.size()

    w_min, w_max = 0.1, 0.9
    img_cut_ratio = w_min + torch.rand(1).item() * (w_max - w_min)
    img_cut_index = int(W * img_cut_ratio)
    point_cut_ratio = (0.5 - img_cut_ratio) * 2 * 0.30769

    indices = torch.randperm(B).to(DEVICE)

    rgbs = torch.cat([rgbs[..., :img_cut_index], rgbs[indices, ..., img_cut_index:]], dim=3)
    silhouettes = torch.cat([silhouettes[..., :img_cut_index], silhouettes[indices, ..., img_cut_index:]], dim=3)

    view_center_points = torch.cat([view_center_points[view_center_points[..., 2] > point_cut_ratio],
                                    view_center_points[view_center_points[indices, :, 2] <= point_cut_ratio]], dim=1)

    return rgbs, silhouettes, view_center_points


def check_parameters(rgbs: torch.Tensor, silhouettes: torch.Tensor, view_center_points: torch.Tensor):
    assert rgbs.ndimension() == silhouettes.ndimension() == 4  # (B, C, H, W)
    assert view_center_points.ndimension() == 3  # (B, N, 3)
