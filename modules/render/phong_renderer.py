import torch
from kaolin.rep import TriangleMesh
from kaolin.graphics import DIBRenderer
from config import DEVICE


renderer = DIBRenderer(128, 128, mode='Phong')
material = torch.tensor([[[0.7, 0.7, 0.7], [0.9, 0.9, 0.9], [0.3, 0.3, 0.3]]], dtype=torch.float).to(DEVICE)
light = torch.tensor([[0, 10, -10]], dtype=torch.float).to(DEVICE)
shininess = torch.tensor([1], dtype=torch.float).to(DEVICE)


class PhongRenderer:
    def __init__(self):
        pass

    @classmethod
    def render(cls, mesh: TriangleMesh, dist, elev, azim, uv=None, texture=None):
        cls.check_mesh_parameters(mesh, uv, texture)
        dist, elev, azim = cls.check_camera_parameters(dist, elev, azim)

        renderer.set_look_at_parameters([azim], [elev], [dist])

        vertices = mesh.vertices.clone().to(DEVICE)[None]
        faces = mesh.faces.clone().to(DEVICE)

        uv, texture = cls.get_random_color(vertices.size(1)) if uv is None and texture is None else uv, texture

        render_imgs, render_alphas, face_norms = renderer.forward(points=[vertices, faces],
                                                                  uv_bxpx2=uv,
                                                                  texture_bx3xthxtw=texture,
                                                                  lightdirect_bx3=light,
                                                                  material_bx3x3=material,
                                                                  shininess_bx1=shininess)
        return render_imgs, render_alphas, face_norms

    @staticmethod
    def check_camera_parameters(dist, elev, azim):
        if isinstance(dist, torch.Tensor):
            dist = dist.item()
        if isinstance(elev, torch.Tensor):
            elev = elev.item()
        if isinstance(azim, torch.Tensor):
            azim = azim.item()
        return dist, elev, azim

    @staticmethod
    def check_mesh_parameters(mesh: TriangleMesh, uv: torch.Tensor, texture: torch.Tensor):
        assert isinstance(mesh, TriangleMesh)
        if uv is None and texture is None:
            return

        assert uv.ndimension() == 3  # (B, N, 2)
        assert texture.ndimension() == 4  # (B, 3, TH, TW)
        assert uv.size(1) == mesh.vertices.size(0) and uv.size(2) == 2
        assert texture.size(1) == 3

    @staticmethod
    def get_random_color(vertex_num: int) -> (torch.Tensor, torch.Tensor):
        uv = torch.rand((1, vertex_num, 2)).to(DEVICE)
        texture = torch.rand((1, 3, 1, 1)).to(DEVICE) * 255

        return uv, texture
