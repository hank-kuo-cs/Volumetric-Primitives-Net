import torch
from PIL import Image
from kaolin.rep import TriangleMesh
from .render import render, to_pil, concat_pil_image
from ..augmentation.acd import merge_meshes

COLORS = torch.tensor([[0.0, 0.99, 0.0], [0.0, 0.0, 0.99], [0.99, 0.0, 0.0], [0.6, 0.3, 0.8],
                       [0.99, 0.99, 0.0], [0.0, 0.99, 0.99], [0.99, 0.0, 0.99], [0.6, 0.8, 0.3],
                       [0.3, 0.6, 0.8], [0.3, 0.8, 0.6], [0.8, 0.3, 0.6], [0.8, 0.6, 0.3],
                       [0.8, 0.1, 0.2], [0.8, 0.2, 0.1], [0.2, 0.1, 0.8], [0.2, 0.8, 0.1],
                       [0.2, 0.1, 0.3], [0.4, 0.6, 0.5], [0.9, 0.9, 0.8], [0.7, 0.99, 0.99]])


def visualize_vp_meshes_with_gif(image: torch.Tensor, vp_meshes: list, save_name: str, dist=2.0, is_three_elev=False):
    image = to_pil(image.cpu()).resize((256, 256), Image.BILINEAR)
    colors = get_colors_of_vps(vp_meshes)
    mesh = compose_vp_meshes(vp_meshes)

    vertices = mesh.vertices[None]
    faces = mesh.faces
    elevs = [-30, 0, 30] if is_three_elev else [0]

    gif_imgs = []
    render_img_direct_pose = render(vertices, faces, colors, dist, 0, 0)
    for azim in range(0, 360, 30):
        imgs = [image, render_img_direct_pose]
        for elev in elevs:
            imgs.append(render(vertices, faces, colors, dist, elev, azim))

        gif_imgs.append(concat_pil_image(imgs))

    gif_imgs[0].save(save_name, format='GIF', append_images=gif_imgs[1:], save_all=True, duration=300, loop=0)


def get_colors_of_vps(vp_meshes: list):
    colors = []

    for i in range(len(vp_meshes)):
        vertex_num = vp_meshes[i].vertices.size(0)

        colors.append(torch.cat(
            [torch.full(size=(vertex_num, 1), fill_value=COLORS[i, j].item()) for j in range(3)], dim=1))

    colors = torch.cat(colors)[None].cuda()  # (1, N, 3)
    return colors


def compose_vp_meshes(vp_meshes: list):
    vertices = []
    faces = []

    last_vertices_num = 0

    for i in range(len(vp_meshes)):
        vp_vertices = vp_meshes[i].vertices
        vertices.append(vp_vertices)

        vp_faces = vp_meshes[i].faces
        vp_faces += last_vertices_num
        faces.append(vp_faces)

        last_vertices_num += vp_vertices.size(0)

    vertices = torch.cat(vertices)
    faces = torch.cat(faces)

    mesh = TriangleMesh.from_tensors(vertices=vertices, faces=faces)
    mesh.cuda()

    return mesh


def visualize_refine_vp_meshes(image: torch.Tensor, vp_meshes: list, save_name: str, predict_vertices: torch.Tensor):
    image = to_pil(image.cpu()).resize((256, 256), Image.BILINEAR)
    colors = get_colors_of_vps(vp_meshes)
    mesh = compose_vp_meshes(vp_meshes)
    mesh.vertices = predict_vertices

    union_mesh = merge_meshes(vp_meshes)
    single_colors = torch.full_like(union_mesh.vertices, fill_value=0.5)[None]

    vertices = mesh.vertices[None]
    faces = mesh.faces

    gif_imgs = []
    render_img_direct_pose = render(vertices, faces, colors, 1, 0, 0)
    for azim in range(0, 360, 30):
        vp_img = render(vertices, faces, colors, 1, 0, azim)
        single_color_img = render(union_mesh.vertices[None], union_mesh.faces, single_colors, 1, 0, azim)

        imgs = [image, render_img_direct_pose, vp_img, single_color_img]
        gif_imgs.append(concat_pil_image(imgs))

    gif_imgs[0].save(save_name, format='GIF', append_images=gif_imgs[1:], save_all=True, duration=300, loop=0)
