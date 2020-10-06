import torch
from PIL import Image
from kaolin.rep import TriangleMesh
from .render import render, to_pil, concat_pil_image


def visualize_mesh_with_gif(image: torch.Tensor, mesh: TriangleMesh, save_name: str):
    image = to_pil(image.cpu()).resize((256, 256), Image.BILINEAR)

    vertices = mesh.vertices[None]
    colors = torch.full_like(vertices, fill_value=0.5).cuda()
    faces = mesh.faces

    gif_imgs = []
    for azim in range(0, 360, 30):
        imgs = [image]
        for elev in range(-30, 60, 30):
            imgs.append(render(vertices, faces, colors, 2.0, elev, azim))

        gif_imgs.append(concat_pil_image(imgs))

    gif_imgs[0].save(save_name, format='GIF', append_images=gif_imgs[1:], save_all=True, duration=300, loop=0)


def visualize_mesh_with_3pose(image: torch.Tensor, mesh: TriangleMesh, save_name: str,
                              dist: float, elev: float, azim: float):
    image = to_pil(image.cpu()).resize((256, 256), Image.BILINEAR)

    vertices = mesh.vertices[None]
    colors = torch.full_like(vertices, fill_value=0.5).cuda()
    faces = mesh.faces

    imgs = [image]

    for i in range(3):
        now_azim = (azim + i * 30) % 360
        imgs.append(render(vertices, faces, colors, dist, elev, now_azim))

    img = concat_pil_image(imgs)
    img.save(save_name)
