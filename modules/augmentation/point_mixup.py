import torch
from kaolin.rep import TriangleMesh
from ..loss import EarthMoverDistanceLoss
from ..render import PhongRenderer, VertexRenderer
from ..meshing import ball_pivot_surface_reconstruction, approximate_convex_decomposition
from config import DEVICE


emd_loss_fnc = EarthMoverDistanceLoss()


def point_mixup_data(view_center_points: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    check_parameters(view_center_points)

    mixed_points = mixup_points(view_center_points)
    recon_meshes, colors = points_to_meshes_and_colors(mixed_points)

    rgbs, silhouettes = meshes_to_imgs(recon_meshes, colors)

    return rgbs, silhouettes, mixed_points


def mixup_points(points: torch.Tensor) -> torch.Tensor:
    B, N, _ = points.size()
    mixup_ratio = torch.rand(1).item()
    indices = torch.randperm(B).to(DEVICE)

    mix_batch_points = torch.zeros_like(points).to(DEVICE)

    for b in range(B):
        points1 = points[b]
        points2 = points[indices[b]]

        dist, assignment = emd_loss_fnc(points1[None], points2[None], 0.005, 100)
        points2_match = points2[assignment[0].long()]
        mixed_points = (1 - mixup_ratio) * points1 + mixup_ratio * points2_match
        mix_batch_points[b] = mixed_points

    return mix_batch_points


def points_to_meshes_and_colors(points: torch.Tensor) -> (list, list):
    check_parameters(points)
    broken_meshes = [ball_pivot_surface_reconstruction(points[b]) for b in range(points.size(0))]

    meshes, colors = [], []

    for broken_mesh in broken_meshes:
        mesh, color = approximate_convex_decomposition(broken_mesh)
        meshes.append(mesh)
        colors.append(color)

    return meshes, colors


def meshes_to_imgs(meshes: list, colors: list) -> (torch.Tensor, torch.Tensor):
    rgbs, silhouettes = [], []

    for i in range(len(meshes)):
        p = torch.rand(1).item()
        rgb, silhouette, _ = VertexRenderer.render(meshes[i], 1, 0, 0, colors[i][None]) if p >= 0.5 \
            else PhongRenderer.render(meshes[i], 1, 0, 0)

        rgbs.append(rgb.permute(0, 3, 1, 2))
        silhouettes.append(silhouette.permute(0, 3, 1, 2))

    rgbs = torch.cat(rgbs)
    silhouettes = torch.cat(silhouettes)

    return rgbs, silhouettes


def check_parameters(view_center_points: torch.Tensor):
    assert view_center_points.ndimension() == 3  # (B, N, 3)
    assert view_center_points.size(-1) == 3


def generate_point_mixup_data(view_center_points: torch.Tensor) -> (torch.Tensor, list):
    check_parameters(view_center_points)

    mixed_points = mixup_points(view_center_points)
    recon_meshes, colors = points_to_meshes_and_colors(mixed_points)

    rgbs, silhouettes = meshes_to_imgs(recon_meshes, colors)

    return rgbs, silhouettes, recon_meshes
