import torch
from PIL import Image
from torchvision.transforms import transforms
from kaolin.rep import TriangleMesh
from kaolin.graphics.DIBRenderer import DIBRenderer

renderer = DIBRenderer(256, 256)
to_pil = transforms.ToPILImage()
COLORS = torch.tensor([[0.0, 0.99, 0.0], [0.0, 0.0, 0.99], [0.99, 0.0, 0.0], [0.6, 0.3, 0.8],
                       [0.99, 0.99, 0.0], [0.0, 0.99, 0.99], [0.99, 0.0, 0.99], [0.6, 0.8, 0.3],
                       [0.3, 0.6, 0.8], [0.3, 0.8, 0.6], [0.8, 0.3, 0.6], [0.8, 0.6, 0.3],
                       [0.8, 0.1, 0.2], [0.8, 0.2, 0.1], [0.2, 0.1, 0.8], [0.2, 0.8, 0.1],
                       [0.2, 0.1, 0.3], [0.4, 0.6, 0.5], [0.9, 0.9, 0.8], [0.7, 0.99, 0.99]])


def visualize_mesh_with_gif(image: torch.Tensor, vp_meshes: list, save_name: str):
    image = to_pil(image.cpu()).resize((256, 256), Image.BILINEAR)
    colors = get_colors_of_vps(vp_meshes)
    mesh = compose_vp_meshes(vp_meshes)

    vertices = mesh.vertices[None]
    faces = mesh.faces

    gif_imgs = []
    for azim in range(0, 360, 30):
        imgs = [image]
        for elev in range(-30, 60, 30):
            imgs.append(render(vertices, faces, colors, 2.0, elev, azim))

        gif_imgs.append(concat_pil_image(imgs))

    gif_imgs[0].save(save_name, format='GIF', append_images=gif_imgs[1:], save_all=True, duration=300, loop=0)


def concat_pil_image(imgs: list) -> Image:
    w, h = imgs[0].width, imgs[0].height
    result = Image.new('RGB', (w * len(imgs), h))

    for i, img in enumerate(imgs):
        result.paste(img, (i * w, 0))

    return result


def render(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, dist: float, elev: float, azim: float):
    assert vertices.ndimension() == 3  # (B, N, 3)
    assert faces.ndimension() == 2  # (K, 3)
    assert colors.size() == vertices.size()  # (B, N, 3)

    renderer.set_look_at_parameters([azim], [elev], [dist])
    render_img, _, _ = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

    render_img = render_img.detach().cpu().squeeze(0).permute(2, 0, 1)
    render_img = to_pil(render_img)

    return render_img


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
