import torch
from kaolin.rep import TriangleMesh
from .vp_mesh import visualize_vp_meshes_with_gif
from .mesh import visualize_mesh_with_gif


class Visualizer:
    @staticmethod
    def render_vp_meshes(image: torch.Tensor, vp_meshes: list, save_name: str):
        visualize_vp_meshes_with_gif(image, vp_meshes, save_name)

    @staticmethod
    def render_mesh(image: torch.Tensor, mesh: TriangleMesh, save_name: str):
        visualize_mesh_with_gif(image, mesh, save_name)
