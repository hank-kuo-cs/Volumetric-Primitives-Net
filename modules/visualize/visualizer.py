import torch
from kaolin.rep import TriangleMesh
from .vp_mesh import visualize_vp_meshes_with_gif, visualize_refine_vp_meshes
from .mesh import visualize_mesh_with_gif, visualize_mesh_with_3pose
from .depth import save_depth_imgs


class Visualizer:
    @staticmethod
    def render_vp_meshes(image: torch.Tensor, depth: torch.Tensor, vp_meshes: list, save_name: str,
                         dist: float = 2.0, is_three_elev: bool = False):
        visualize_vp_meshes_with_gif(image, depth, vp_meshes, save_name, dist=dist, is_three_elev=is_three_elev)

    @staticmethod
    def render_refine_vp_meshes(image: torch.Tensor, vp_meshes: list, predict_vertices: torch.Tensor, save_name: str):
        visualize_refine_vp_meshes(image, vp_meshes, save_name, predict_vertices)


    @staticmethod
    def render_mesh_gif(image: torch.Tensor, mesh: TriangleMesh, save_name: str, dist: float):
        visualize_mesh_with_gif(image, mesh, save_name, dist)

    @staticmethod
    def render_mesh_3pose(image: torch.Tensor, mesh: TriangleMesh, save_name: str,
                          dist: float, elev: float, azim: float):
        visualize_mesh_with_3pose(image, mesh, save_name, dist, elev, azim)

    @staticmethod
    def save_depth_imgs(predict_depth, gt_depth, save_path):
        save_depth_imgs(predict_depth, gt_depth, save_path)
