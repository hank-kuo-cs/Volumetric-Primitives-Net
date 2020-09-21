import torch
from kaolin.graphics import DIBRenderer
from config import DEVICE


renderer = DIBRenderer(128, 128)


class Renderer:
    def __init__(self):
        pass

    @staticmethod
    def render(mesh, dist, elev, azim):
        renderer.set_look_at_parameters([azim], [elev], [dist])

        vertices = mesh.vertices.clone().to(DEVICE)[None]
        faces = mesh.faces.clone().to(DEVICE)
        colors = torch.ones_like(vertices).to(DEVICE)

        render_rgb, render_alpha, face_norms = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

        return render_rgb, render_alpha, face_norms
