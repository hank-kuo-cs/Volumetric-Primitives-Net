import os
import re
import torch
from tqdm import tqdm
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from kaolin.rep import TriangleMesh
from .data import ShapeNetData
from config import *


class_dict = {'airplane': '02691156', 'rifle': '04090263', 'display': '03211117', 'table': '04379243',
              'telephone': '04401088', 'car': '02958343', 'chair': '03001627', 'bench': '02828884', 'lamp': '03636649',
              'cabinet': '02933112', 'loudspeaker': '03691459', 'sofa': '04256520', 'watercraft': '04530566'}

img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_type: str):
        assert dataset_type == 'train' or dataset_type == 'test'
        self.dataset_type = dataset_type
        self.shapenet_datas = []
        self.split_data = {}

        self._load_data()

    def __len__(self) -> int:
        return len(self.shapenet_datas)

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        shapenet_data = self.shapenet_datas[item]
        dist, elev, azim = shapenet_data.dist, shapenet_data.elev, shapenet_data.azim
        rgb, silhouette = self._load_rgb_and_silhouette(shapenet_data.img_path)

        # vertices, faces = self._load_mesh(shapenet_data.obj_path)
        # vertices = self.transform_to_view_center(vertices, shapenet_data.elev, shapenet_data.azim)

        points = self._load_sample_points(shapenet_data.obj_path)

        return rgb, silhouette, points, dist, elev, azim

    def _load_data(self):
        self._load_split_data()
        CLASSES = TRAIN_CLASSES if self.dataset_type == 'train' else TEST_CLASSES

        dataset_indices = self.split_data['train'] if self.dataset_type == 'train' else self.split_data['test']

        class_ids = [class_dict[class_name] for class_name in CLASSES]
        class_model_num = [0 for class_name in CLASSES]

        for i in tqdm(range(len(dataset_indices))):
            class_id, obj_id = dataset_indices[i][0], dataset_indices[i][1]
            if class_id not in class_ids:
                continue

            class_index = class_ids.index(class_id)
            if LITTLE_NUM[self.dataset_type] > 0 and class_model_num[class_index] >= LITTLE_NUM[self.dataset_type] // len(CLASSES):
                continue
            class_model_num[class_index] += 1

            imgs_dir_path = os.path.join(DATASET_ROOT, 'ShapeNetRendering', class_id, obj_id, 'rendering')

            imgs_path, azims, elevs, dists = self._load_imgs_in_one_dir(imgs_dir_path)
            obj_path = os.path.join(DATASET_ROOT, 'ShapeNetCore.v1', class_id, obj_id, 'model.obj')

            for j in range(len(imgs_path)):
                shapenet_data = ShapeNetData(img_path=imgs_path[j], obj_path=obj_path,
                                             dist=dists[j], elev=elevs[j], azim=azims[j])
                self.shapenet_datas.append(shapenet_data)

        for i in range(len(class_model_num)):
            print(CLASSES[i], class_model_num[i])

    def _load_split_data(self):
        split_data = {}
        str_data = open(DATASET_ROOT + '/split.csv', 'r').read()

        split_data['train'] = re.findall(r'.+,(.+),.+,(.+),train', str_data)
        split_data['test'] = re.findall(r'.+,(.+),.+,(.+),test', str_data)
        split_data['train'].extend(re.findall(r'.+,(.+),.+,(.+),val', str_data))

        self.split_data = split_data

    def _load_imgs_in_one_dir(self, dir_path):
        meta_path = dir_path + '/rendering_metadata.txt'
        imgs_path = sorted(glob(dir_path + '/*.png'))
        azims, elevs, dists = self._load_meta(meta_path)

        return imgs_path, azims, elevs, dists

    @staticmethod
    def _load_rgb_and_silhouette(img_path: str) -> (torch.Tensor, torch.Tensor):
        img = Image.open(img_path).convert('RGB')
        img = img_transform(img)

        return img[:3], img[3]

    @staticmethod
    def _load_meta(meta_path: str) -> (list, list, list):
        azims, elevs, dists = [], [], []
        meta_str = open(meta_path, 'r').read()
        meta_datas = re.findall(r'([\.0-9]+?) ([\.0-9]+?) 0 ([\.0-9]+?) 25', meta_str)
        for meta_data in meta_datas:
            cameras = list(map(float, meta_data))
            cameras[2] *= 1.754

            azims.append(cameras[0])
            elevs.append(cameras[1])
            dists.append(cameras[2])

        return azims, elevs, dists

    @staticmethod
    def _load_mesh(obj_path: str) -> (torch.Tensor, torch.Tensor):
        mesh = TriangleMesh.from_obj(obj_path)
        return mesh.vertices, mesh.faces

    @staticmethod
    def _load_sample_points(obj_path: str) -> torch.Tensor:
        mesh = TriangleMesh.from_obj(obj_path)
        return mesh.sample(SAMPLE_NUM * vp_num)[0]

    @staticmethod
    def transform_to_view_center(vertices: torch.Tensor, elev: float, azim: float) -> torch.Tensor:
        return vertices
        # TODO: normalize vertices to view-centered coordinate
        #
        # assert vertices.size(-1) == 3 and vertices.ndimension() == 2
        # normalize_vertices = None
        #
        #
        #
        # assert normalize_vertices.size(-1) == 3 and normalize_vertices.ndimension() == 2
        # return normalize_vertices
