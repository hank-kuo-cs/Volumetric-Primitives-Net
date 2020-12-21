import torch
from torchvision.transforms import transforms
from .render import concat_pil_image


def save_depth_imgs(predict_depth: torch.Tensor, gt_depth: torch.Tensor, save_path: str):
    assert predict_depth.ndimension() == 3  # (1, H, W)
    assert gt_depth.ndimension() == 3  # (1, H, W)

    predict_depth = predict_depth.detach().cpu()
    gt_depth = gt_depth.detach().cpu()

    predict_depth = transforms.ToPILImage()(predict_depth)
    gt_depth = transforms.ToPILImage()(gt_depth)

    img = concat_pil_image([predict_depth, gt_depth])
    img.save(save_path)
