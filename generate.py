import os
import numpy as np
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from kaolin.rep import TriangleMesh
from modules import ShapeNetDataset
from modules.augmentation.point_mixup import generate_point_mixup_data
from modules.meshing.convex_decomposition import get_kaolinmesh_from_trimesh, get_trimesh_from_kaolinmesh
from modules.render import PhongRenderer
from modules.transform import rotate_points_forward_x_axis
from config import *


to_pil = transforms.ToPILImage()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_device', type=str, default='0', help='gpu device number')
    parser.add_argument('-d', '--dataset', type=str, help='point_mixup, acd')
    parser.add_argument('-p', '--path', type=str, help='dataset path')

    return parser.parse_args()


def load_dataloader(data_type):
    print('Load %s dataset...' % data_type)
    dataset = ShapeNetDataset(data_type)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    print('Dataset size =', len(dataset))

    return dataloader


def generate_point_mixup_dataset(dataset_path):
    dataloader = load_dataloader('train')
    progress_bar = tqdm(dataloader)

    n = 1

    for epoch in range(1):
        for data in progress_bar:
            rotate_angles = data['rotate_angle'].float().to(DEVICE)
            view_center_points = data['view_center_points'].to(DEVICE)

            # view_center_points = rotate_points_forward_x_axis(view_center_points, rotate_angles)
            rgbs, silhouettes, meshes = generate_point_mixup_data(view_center_points)

            for i in range(rgbs.size(0)):
                rgb = to_pil(rgbs[i].cpu())
                silhouette = to_pil(silhouettes[i].cpu())

                rgb.save(os.path.join(dataset_path, 'rgb_%d.png' % n))
                silhouette.save(os.path.join(dataset_path, 'silhouette_%d.png' % n))

                mesh = meshes[i]
                mesh.save_mesh(os.path.join(dataset_path, 'mesh_%d.obj' % n))

                n += 1


def generate_acd_dataset(dataset_path: str):
    dataset = ShapeNetDataset('train')

    n = 0

    for data in dataset.shapenet_datas:
        mesh = TriangleMesh.from_obj(data.canonical_obj_path)
        convex_hulls = get_trimesh_from_kaolinmesh(mesh).convex_decomposition(6)

        if len(convex_hulls) != 6:
            print('convex hull num != 6, obj path =', data.canonical_obj_path)

        vertices, faces = [], []

        for convex_hull in convex_hulls:
            vertices.append(np.array(convex_hull.vertices).tolist())
            faces.append(np.array(convex_hull.faces).tolist())

        json_data = json.dumps({'vertices': vertices, 'faces': faces})
        with open(os.path.join(dataset_path, 'mesh_%.6d.json' % n), 'w') as f:
            f.write(json_data)
            f.close()

        n += 1


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    dataset_path = args.path
    os.makedirs(dataset_path, exist_ok=True)

    if args.dataset == 'point_mixup':
        generate_point_mixup_dataset(dataset_path)
    elif args.dataset == 'acd':
        generate_acd_dataset(dataset_path)


