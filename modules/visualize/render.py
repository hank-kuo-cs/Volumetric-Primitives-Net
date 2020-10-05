import torch
from PIL import Image
from torchvision.transforms import transforms
from kaolin.graphics.DIBRenderer import DIBRenderer


renderer = DIBRenderer(256, 256)
to_pil = transforms.ToPILImage()


def render(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, dist: float, elev: float, azim: float):
    assert vertices.ndimension() == 3  # (B, N, 3)
    assert faces.ndimension() == 2  # (K, 3)
    assert colors.size() == vertices.size()  # (B, N, 3)

    renderer.set_look_at_parameters([azim], [elev], [dist])
    render_img, _, _ = renderer.forward(points=[vertices, faces], colors_bxpx3=colors)

    render_img = render_img.detach().cpu().squeeze(0).permute(2, 0, 1)
    render_img = to_pil(render_img)

    return render_img


def concat_pil_image(imgs: list) -> Image:
    w, h = imgs[0].width, imgs[0].height
    result = Image.new('RGB', (w * len(imgs), h))

    for i, img in enumerate(imgs):
        result.paste(img, (i * w, 0))

    return result



