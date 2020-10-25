import torch
from ..transform import rotate_points


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
