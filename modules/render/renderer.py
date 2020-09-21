import torch
from kaolin.rep import TriangleMesh
from kaolin.graphics import DIBRenderer
from config import DEVICE


renderer = DIBRenderer(128, 128)


class Renderer:
    def __init__(self):
        pass

    @classmethod
    def render(cls, mesh, dist, elev, azim):
        isinstance(mesh, TriangleMesh)
        dist, elev, azim = cls.check_camera_parameters(dist, elev, azim)
        renderer.set_look_at_parameters([azim], [elev], [dist])

        vertices = mesh.vertices.clone().to(DEVICE)[None]
        faces = mesh.faces.clone().to(DEVICE)
        colors = torch.ones_like(vertices).to(DEVICE)

        render_rgb, render_alpha, face_norms = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

        return render_rgb, render_alpha, face_norms

    @staticmethod
    def check_camera_parameters(dist, elev, azim):
        if isinstance(dist, torch.Tensor):
            dist = dist.item()
        if isinstance(elev, torch.Tensor):
            elev = elev.item()
        if isinstance(azim, torch.Tensor):
            azim = azim.item()
        return dist, elev, azim
