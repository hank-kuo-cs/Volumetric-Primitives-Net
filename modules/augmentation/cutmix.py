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

    view_center_points = cut_mix_batch_points(view_center_points, indices, point_cut_ratio)

    return rgbs, silhouettes, view_center_points


def cut_mix_batch_points(view_center_points, indices, cut_ratio):
    B, N = view_center_points.size(0), view_center_points.size(1)
    mix_batch_points = torch.zeros_like(view_center_points).to(DEVICE)

    for b in range(B):
        points1 = view_center_points[b]
        points2 = view_center_points[indices[b]]

        mix_points = torch.cat([points1[points1[:, 2] >= cut_ratio], points2[points2[:, 2] < cut_ratio]], dim=0)
        mix_batch_points[b] = adjust_point_num(mix_points, N)

    return mix_batch_points


def adjust_point_num(points: torch.Tensor, N: int):
    assert points.ndimension() == 2  # (N', 3)

    if points.size(0) == N:
        return points
    elif points.size(0) > N:
        indices = torch.randperm(points.size(0))[:N].to(DEVICE)
        return points[indices, :]
    else:
        indices = torch.randint(0, points.size(0), (N,)).to(DEVICE)
        return points[indices, :]


def check_parameters(rgbs: torch.Tensor, silhouettes: torch.Tensor, view_center_points: torch.Tensor):
    assert rgbs.ndimension() == silhouettes.ndimension() == 4  # (B, C, H, W)
    assert view_center_points.ndimension() == 3  # (B, N, 3)
