import torch
from .render import visualize_mesh_with_gif


class Visualizer:
    @staticmethod
    def render_vp_meshes(image: torch.Tensor, vp_meshes: list, save_name: str):
        visualize_mesh_with_gif(image, vp_meshes, save_name)
