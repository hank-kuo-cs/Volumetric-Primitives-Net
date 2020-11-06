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
from .classes import Classes
from ..transform.rotate import rotate_points
from config import *

img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
    transforms.ToTensor()
])

vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_type: str):
        assert dataset_type == 'train' or dataset_type == 'test_seen' or dataset_type == 'test_unseen'
        self.dataset_type = dataset_type
        self.shapenet_datas = []
        self.split_data = {}

        self._load_data()

    def __len__(self) -> int:
        return len(self.shapenet_datas)

    def __getitem__(self, item) -> dict:
        shapenet_data = self.shapenet_datas[item]

        dist, elev, azim = shapenet_data.dist, shapenet_data.elev, shapenet_data.azim
        rgb, silhouette, rotate_angle = self._load_rgb_and_silhouette(shapenet_data.img_path)

        canonical_points = self._load_sample_points(shapenet_data.canonical_obj_path)
        view_center_points = self._load_sample_points(shapenet_data.view_center_obj_path)

        if IS_DIST_INVARIANT:
            view_center_points *= dist

        return {
            'rgb': rgb,
            'silhouette': silhouette,
            'canonical_points': canonical_points,
            'view_center_points': view_center_points,
            'class_index': shapenet_data.class_index,
            'rotate_angle': rotate_angle,
            'dist': dist,
            'elev': elev,
            'azim': azim
        }

    def _load_data(self):
        self._load_split_data()
        CLASSES = {'train': TRAIN_CLASSES,
                   'test_seen': TEST_SEEN_CLASSES,
                   'test_unseen': TEST_UNSEEN_CLASSES}[self.dataset_type]

        dataset_indices = self.split_data['train'] if self.dataset_type == 'train' else self.split_data['test']

        class_ids = [Classes.get_id_by_name(class_name) for class_name in CLASSES]
        class_model_num = [0 for class_name in CLASSES]

        for i in tqdm(range(len(dataset_indices))):
            class_id, obj_id = dataset_indices[i][0], dataset_indices[i][1]
            if class_id not in class_ids:
                continue

            class_index = class_ids.index(class_id)
            if 0 < LITTLE_NUM[self.dataset_type] <= class_model_num[class_index]:
                continue
            class_model_num[class_index] += 1

            imgs_dir_path = os.path.join(DATASET_ROOT, 'ShapeNetRendering', class_id, obj_id, 'rendering')

            imgs_path, azims, elevs, dists = self._load_imgs_in_one_dir(imgs_dir_path)
            object_center_obj_path = os.path.join(DATASET_ROOT, 'ShapeNetCore.v1', class_id, obj_id, 'model.obj')
            view_center_objs_path = sorted(glob(os.path.join(DATASET_ROOT, 'ShapeNetRendering', class_id, obj_id, 'objs') + '/*.obj'))

            for j in range(len(imgs_path)):
                shapenet_data = ShapeNetData(img_path=imgs_path[j],
                                             canonical_obj_path=object_center_obj_path,
                                             view_center_obj_path=view_center_objs_path[j],
                                             class_index=Classes.get_class_index_by_id(class_id),
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

    def _load_rgb_and_silhouette(self, img_path: str) -> (torch.Tensor, torch.Tensor):
        angle = 0.0
        img = Image.open(img_path)
        img = img_transform(img)

        if AUGMENT_3D['rotate'] and self.dataset_type == 'train':
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
    def _load_mesh(obj_path: str) -> TriangleMesh:
        mesh = TriangleMesh.from_obj(obj_path)
        return mesh

    @staticmethod
    def _load_sample_points(obj_path: str) -> torch.Tensor:
        mesh = TriangleMesh.from_obj(obj_path)
        # return mesh.sample(SAMPLE_NUM * vp_num)[0]
        return mesh.sample(2048)[0]

    @staticmethod
    def transform_to_view_center(vertices: torch.Tensor, dist: float, elev: float, azim: float) -> torch.Tensor:
        assert vertices.size(-1) == 3 and vertices.ndimension() == 2
        y_vec, neg_z_vec = [0, 1, 0], [0, 0, -1]

        q = torch.tensor([*neg_z_vec, elev / 360], dtype=torch.float)[None].to(DEVICE)
        vertices = vertices[None].to(DEVICE)
        vertices = rotate_points(vertices, q)

        y_vec_tensor = torch.tensor(y_vec, dtype=torch.float)[None, None].to(DEVICE)
        y_vec = rotate_points(y_vec_tensor, q)[0, 0].tolist()

        q = torch.tensor([*y_vec, azim / 360], dtype=torch.float)[None].to(DEVICE)
        vertices = rotate_points(vertices, q)

        vertices = vertices.squeeze(0) / dist

        return vertices.detach().cpu()

    def save_view_center_dataset(self):
        for i in tqdm(range(len(self.shapenet_datas))):
            shapenet_data = self.shapenet_datas[i]
            img_path, obj_path = shapenet_data.img_path, shapenet_data.obj_path
            dist, elev, azim = shapenet_data.dist, shapenet_data.elev, shapenet_data.azim

            mesh = self._load_mesh(obj_path)
            mesh.vertices = self.transform_to_view_center(mesh.vertices, dist, elev, azim)

            new_obj_path = re.sub('rendering', 'objs', img_path)
            new_obj_path = re.sub('png', 'obj', new_obj_path)

            os.makedirs(os.path.dirname(new_obj_path), exist_ok=True)

            mesh.save_mesh(new_obj_path)
