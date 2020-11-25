import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from kaolin.rep import TriangleMesh
from config import *

img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
    transforms.ToTensor()
])

vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM


class PointMixUpDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_path = os.path.join(DATASET_ROOT, dataset_name)
        self.rgb_paths = sorted(glob(self.dataset_path + '/rgb*.png'))
        self.silhouette_paths = sorted(glob(self.dataset_path + '/silhouette*.png'))
        self.obj_paths = sorted(glob(self.dataset_path + '/mesh*.obj'))

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, item) -> dict:
        rgb_path = self.rgb_paths[item]
        silhouette_path = self.silhouette_paths[item]
        obj_path = self.obj_paths[item]

        rgb = img_transform(Image.open(rgb_path))
        silhouette = img_transform(Image.open(silhouette_path))
        points = TriangleMesh.from_obj(obj_path).sample(2048)[0]
        angle = 0.0

        if AUGMENT_3D['rotate']:
            rgb, silhouette, angle = self.rotate_img(rgb, silhouette)

        return {
            'rgb': rgb,
            'silhouette': silhouette,
            'points': points,
            'angle': angle,
        }

    @staticmethod
    def rotate_img(rgb: torch.Tensor, silhouette: torch.Tensor) -> (torch.Tensor, torch.Tensor, float):
        angle = torch.rand(1) * 360
        rotate_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((angle, angle)),
            transforms.ToTensor()
        ])

        return rotate_transform(rgb), rotate_transform(silhouette), angle.item()
