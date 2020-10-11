import torch
from kaolin.rep import TriangleMesh
from .vp_mesh import visualize_vp_meshes_with_gif
from .mesh import visualize_mesh_with_gif, visualize_mesh_with_3pose


class Visualizer:
    @staticmethod
    def render_vp_meshes(image: torch.Tensor, vp_meshes: list, save_name: str, dist: float = 2.0):
        visualize_vp_meshes_with_gif(image, vp_meshes, save_name, dist=dist)

    @staticmethod
    def render_mesh_gif(image: torch.Tensor, mesh: TriangleMesh, save_name: str):
        visualize_mesh_with_gif(image, mesh, save_name)

    @staticmethod
    def render_mesh_3pose(image: torch.Tensor, mesh: TriangleMesh, save_name: str,
                          dist: float, elev: float, azim: float):
        visualize_mesh_with_3pose(image, mesh, save_name, dist, elev, azim)
