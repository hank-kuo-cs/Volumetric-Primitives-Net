import os
import torch
from glob import glob
from PIL import Image
from kaolin.rep import TriangleMesh
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from .classes import Classes
from config import TEST_UNSEEN_CLASSES, GENRE_TESTING_ROOT, IMG_SIZE


img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])


class GenReDataset(Dataset):
    """
    GenRe (NIPS 2018) testing dataset
    """
    def __init__(self, dataset_path: str = GENRE_TESTING_ROOT):
        super().__init__()
        self.root = dataset_path
        self.rgb_paths = []
        self.mask_paths = []
        self.obj_paths = []
        self.class_indices = []

        self._load_data()
        self._check_path_data()

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, item):
        rgb = self._load_img(self.rgb_paths[item], self.mask_paths[item])
        points = self._load_points(self.obj_paths[item])

        return {'rgb': rgb, 'view_center_points': points, 'class_index': self.class_indices[item]}

    def _load_data(self):
        for class_name in TEST_UNSEEN_CLASSES:
            class_id = Classes.get_id_by_name(class_name)
            obj_paths = glob(os.path.join(self.root, class_id, '*'))

            for obj_path in obj_paths:
                self.rgb_paths += sorted(glob(os.path.join(obj_path, '*rgb.png')))
                self.mask_paths += sorted(glob(os.path.join(obj_path, '*silhouette.png')))
                self.obj_paths += sorted(glob(os.path.join(obj_path, '*.obj')))
                self.class_indices += [Classes.get_class_index_by_id(class_id) for i in range(20)]

    def _check_path_data(self):
        assert len(self.rgb_paths) == len(self.mask_paths) == len(self.obj_paths) == len(self.class_indices)

    @staticmethod
    def _load_img(rgb_path: str, mask_path: str) -> torch.Tensor:
        rgb, mask = Image.open(rgb_path), Image.open(mask_path)
        rgb, mask = img_transform(rgb), img_transform(mask)

        mask = torch.where(mask >= 0.5, torch.ones_like(mask), torch.zeros_like(mask))
        rgb = rgb * mask

        return rgb

    @staticmethod
    def _load_points(obj_path):
        mesh = TriangleMesh.from_obj(obj_path)
        mesh.vertices -= torch.mean(mesh.vertices, 0)
        mesh.vertices /= 128
        mesh.vertices = mesh.vertices[:, [0, 2, 1]]
        mesh.vertices[:, 0] *= -1
        mesh.vertices /= 1.7
        return mesh.sample(2048)[0]
