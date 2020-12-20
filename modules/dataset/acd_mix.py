import os
import torch
import json
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from ..render import DepthRenderer
from kaolin.rep import TriangleMesh
from config import *

img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
    transforms.ToTensor()
])

vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM


class ACDMixDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = sorted(glob(self.dataset_path + '/img_*.png'))
        self.obj_paths = sorted(glob(self.dataset_path + '/mesh_*.obj'))
        self.meta_paths = sorted(glob(self.dataset_path + '/meta_*.json'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        rgb, silhouette, angle = self.load_img(self.image_paths[item])

        mesh = self.load_mesh(self.obj_paths[item])
        points = self.sample_points(mesh)
        depth = DepthRenderer.render_depth(mesh)

        dist, elev, azim = self.load_meta(self.meta_paths[item])
        return {
            'rgb': rgb,
            'silhouette': silhouette,
            'points': points,
            'rotate_angle': angle,
            'depth': depth,
            'dist': dist,
            'elev': elev,
            'azim': azim
        }

    def load_img(self, img_path: str):
        angle = 0.0
        img = Image.open(img_path)
        img = img_transform(img)

        if AUGMENT_3D['rotate']:
            img, angle = self.rotate_img(img)

        rgb, silhouette = img[:3], img[3].unsqueeze(0)

        if IS_NORMALIZE:
            rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb)

        return rgb, silhouette, angle

    @staticmethod
    def rotate_img(img: torch.Tensor) -> (torch.Tensor, float):
        angle = torch.rand(1) * 360
        rotate_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((angle, angle)),
            transforms.ToTensor()
        ])

        return rotate_transform(img), angle.item()

    @staticmethod
    def load_mesh(obj_path: str):
        return TriangleMesh.from_obj(obj_path)

    @staticmethod
    def sample_points(mesh: TriangleMesh, num: int = 2048):
        return mesh.sample(num)[0]

    @staticmethod
    def load_meta(meta_path: str):
        data = json.load(open(meta_path))
        dist = torch.tensor(data['dist'], dtype=torch.float)
        elev = torch.tensor(data['elev'], dtype=torch.float)
        azim = torch.tensor(data['azim'], dtype=torch.float)
        return dist, elev, azim
