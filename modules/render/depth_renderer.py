"""
I use pytorch3d (https://github.com/facebookresearch/pytorch3d) to render the depth map of mesh.
"""

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, OpenGLPerspectiveCameras, MeshRasterizer


class DepthRenderer:
    def __init__(self):
        pass

    @staticmethod
    def render_depth(mesh, dist=1.0, elev=0, azim=90, device='cpu'):
        mesh = Meshes(verts=[mesh.vertices.float()], faces=[mesh.faces])

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = OpenGLPerspectiveCameras(fov=50, R=R, T=T, device=device)
        raster_settings = RasterizationSettings(image_size=128, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        rendered_depth = rasterizer(mesh).zbuf[0].permute(2, 0, 1)

        depth_indices = rendered_depth >= 0
        non_depth_indices = rendered_depth < 0

        rendered_depth[depth_indices] = rendered_depth[depth_indices].max() - rendered_depth[depth_indices]
        rendered_depth[depth_indices] /= rendered_depth[depth_indices].max()

        rendered_depth[non_depth_indices] = torch.zeros_like(rendered_depth[non_depth_indices])

        return rendered_depth  # (1, 128, 128)

    @classmethod
    def render_batch_depth(cls, meshes):
        return torch.cat([cls.render_depth(mesh, device='cuda')[None] for mesh in meshes])  # (B, 1, 128, 128)
