import torch
from .rotate import rotate_points
from .translate import translate_points


def transform_points(points: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
    check_parameters(points, q, t)

    return translate_points(rotate_points(points, q), t)


def check_parameters(points: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert points.size(-1) == 3
    B = points.size(0)

    assert q.size() == (B, 4)
    assert t.size() == (B, 3)


def view_to_obj_points(points, dists, elevs, azims, angles):
    assert points.ndimension() == 3  # (B, N, 3)
    assert dists.ndimension() == elevs.ndimension() == azims.ndimension() == 1  # (B)
    dists, elevs, azims = dists.view(-1, 1), elevs.view(-1, 1) / 360, azims.view(-1, 1) / 360
    points = rotate_points_forward_x_axis(points, -angles)

    B = points.size(0)
    y = torch.tensor([[0, 1, 0]], device=points.device)
    neg_z = torch.tensor([[0, 0, -1]], device=points.device)
    y = torch.repeat_interleave(y, repeats=B, dim=0).float()
    neg_z = torch.repeat_interleave(neg_z, repeats=B, dim=0).float()

    q1 = torch.cat([neg_z, elevs], dim=1)
    y = rotate_points(y.unsqueeze(1), q1).squeeze(1)

    q2 = torch.cat([y, -azims], dim=1)
    points = rotate_points(points, q2)

    q3 = torch.cat([neg_z, -elevs], dim=1)
    points = rotate_points(points, q3)

    dists = torch.repeat_interleave(dists.unsqueeze(2), repeats=points.size(1), dim=1)
    dists = torch.repeat_interleave(dists, repeats=points.size(2), dim=2)

    points = points * dists

    return points


def obj_to_view_points(points: torch.Tensor, dists: torch.Tensor, elevs: torch.Tensor, azims: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert dists.ndimension() == elevs.ndimension() == azims.ndimension() == 1  # (B)
    dists, elevs, azims = dists.view(-1, 1), elevs.view(-1, 1) / 360, azims.view(-1, 1) / 360

    B = points.size(0)
    y = torch.repeat_interleave(torch.tensor([[0, 1, 0]], device=points.device), repeats=B, dim=0).float()
    neg_z = torch.repeat_interleave(torch.tensor([[0, 0, -1]], device=points.device), repeats=B, dim=0).float()

    q = torch.cat([neg_z, elevs], dim=1)
    points = rotate_points(points, q)

    y = y.unsqueeze(1)
    y = rotate_points(y, q).squeeze(1)

    q = torch.cat([y, azims], dim=1)
    points = rotate_points(points, q)

    dists = torch.repeat_interleave(dists.unsqueeze(2), repeats=points.size(1), dim=1)
    dists = torch.repeat_interleave(dists, repeats=points.size(2), dim=2)

    points = points / dists

    return points


def rotate_points_forward_x_axis(points: torch.Tensor, angles: torch.Tensor):
    """
    Rotate points in view-centered coordinate forward x-axis.
    :param points: (B, N, 3)
    :param angles: (B) value between [0, 360]
    :return:
    """
    assert points.ndimension() == 3
    assert angles.ndimension() == 1

    B = points.size(0)
    x = torch.repeat_interleave(torch.tensor([[1, 0, 0]], device=points.device), repeats=B, dim=0).float()

    angles = angles.view(-1, 1) / 360
    q = torch.cat([x, angles], dim=1)

    points = rotate_points(points, q)

    return points
