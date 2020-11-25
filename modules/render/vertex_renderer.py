import torch
from kaolin.rep import TriangleMesh
from kaolin.graphics import DIBRenderer
from config import DEVICE


renderer = DIBRenderer(128, 128)


class VertexRenderer:
    def __init__(self):
        pass

    @classmethod
    def render(cls, mesh, dist, elev, azim, colors=None):
        isinstance(mesh, TriangleMesh)
        dist, elev, azim = cls.check_camera_parameters(dist, elev, azim)
        renderer.set_look_at_parameters([azim], [elev], [dist])

        vertices = mesh.vertices.clone().to(DEVICE)[None]
        faces = mesh.faces.clone().to(DEVICE)
        colors = torch.ones_like(vertices).to(DEVICE) if colors is None else colors

        render_rgbs, render_alphas, face_norms = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

        return render_rgbs, render_alphas, face_norms

    @staticmethod
    def check_camera_parameters(dist, elev, azim):
        if isinstance(dist, torch.Tensor):
            dist = dist.item()
        if isinstance(elev, torch.Tensor):
            elev = elev.item()
        if isinstance(azim, torch.Tensor):
            azim = azim.item()
        return dist, elev, azim
