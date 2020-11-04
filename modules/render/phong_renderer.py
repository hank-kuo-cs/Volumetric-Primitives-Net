import torch
from kaolin.rep import TriangleMesh
from kaolin.graphics import DIBRenderer
from config import DEVICE


renderer = DIBRenderer(128, 128, mode='Phong')
material = torch.tensor([[[0.1, 0.1, 0.1],[0.9, 0.9, 0.9],[0.5, 0.5, 0.5]]], dtype=torch.float).to(DEVICE)
light = torch.tensor([[0, 10, -10]], dtype=torch.float).to(DEVICE)
shininess = torch.tensor([2], dtype=torch.float).to(DEVICE)


class PhongRenderer:
    def __init__(self):
        pass

    @classmethod
    def render(cls, mesh, dist, elev, azim, color=None):
        isinstance(mesh, TriangleMesh)

        dist, elev, azim = cls.check_camera_parameters(dist, elev, azim)
        renderer.set_look_at_parameters([azim], [elev], [dist])

        material[0, 2, :] = torch.rand(3, dtype=torch.float) * 0.5 + 0.5 if color is None else color

        vertices = mesh.vertices.clone().to(DEVICE)[None]
        faces = mesh.faces.clone().to(DEVICE)

        uv = torch.ones((1, vertices.size(1), 2)).cuda()
        texture = torch.full((1, 3, 128, 128), fill_value=0.5).cuda()

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