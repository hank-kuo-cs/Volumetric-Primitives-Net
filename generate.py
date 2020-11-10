import os
import numpy as np
import json
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from kaolin.rep import TriangleMesh
from modules import ShapeNetDataset
from modules.augmentation.point_mixup import generate_point_mixup_data
from modules.meshing.convex_decomposition import get_kaolinmesh_from_trimesh, get_trimesh_from_kaolinmesh, approximate_convex_decomposition
from modules.render import PhongRenderer
from modules.augmentation.acd import augment, acd, merge_meshes
from config import *


to_pil = transforms.ToPILImage()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_device', type=str, default='0', help='gpu device number')
    parser.add_argument('-d', '--dataset', type=str, help='point_mixup, acd')
    parser.add_argument('-p', '--path', type=str, help='dataset path')
    parser.add_argument('-t', '--type', type=str, help='dataset type')
    parser.add_argument('-n', '--obj_num', type=int, help='dataset size')

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


def generate_acd_dataset(dataset_path: str, data_type='train'):
    dataset = ShapeNetDataset(data_type)

    n = 0
    obj_paths = []

    for data in tqdm(dataset.shapenet_datas):
        if data.canonical_obj_path in obj_paths:
            continue
        obj_paths.append(data.canonical_obj_path)

        try:
            mesh = TriangleMesh.from_obj(data.canonical_obj_path)
            convex_hulls = get_trimesh_from_kaolinmesh(mesh).convex_decomposition(6)
        except Exception as e:
            print('[ACD Exception] obj path = %s, %s' % (data.canonical_obj_path, str(e)))
            continue

        try:
            if len(convex_hulls) != 6:
                print('convex hull num != 6, obj path =', data.canonical_obj_path)
        except Exception as e:
            print('[Hull Num Exception] obj path = %s, %s' % (data.canonical_obj_path, str(e)))
            continue

        vertices, faces = [], []

        for convex_hull in convex_hulls:
            vertices.append(np.array(convex_hull.vertices).tolist())
            faces.append(np.array(convex_hull.faces).tolist())

        json_data = json.dumps({'vertices': vertices, 'faces': faces, 'obj': data.canonical_obj_path})
        with open(os.path.join(dataset_path, 'mesh_%.6d.json' % n), 'w') as f:
            f.write(json_data)
            f.close()

        n += 1


def generate_acd_mix_dataset(dataset_path, args):
    dataset = ShapeNetDataset(args.type)
    obj_paths = []
    dataset_obj_num = args.data_num
    n = 0

    for data in tqdm(dataset.shapenet_datas):
        if data.canonical_obj_path in obj_paths:
            continue
        obj_paths.append(data.canonical_obj_path)

    for i in range(dataset_obj_num):
        rand_two_obj_paths = random.sample(obj_paths, 2)

        obj1_path = rand_two_obj_paths[0]
        obj2_path = rand_two_obj_paths[1]

        k_m1 = TriangleMesh.from_obj(obj1_path)
        k_m2 = TriangleMesh.from_obj(obj2_path)

        k_m1.vertices /= k_m1.vertices.max()
        k_m2.vertices /= k_m2.vertices.max()

        t_m1 = get_trimesh_from_kaolinmesh(k_m1)
        t_m2 = get_trimesh_from_kaolinmesh(k_m2)

        t_hulls1 = acd(t_m1)
        k_hulls1 = [get_kaolinmesh_from_trimesh(t_hull) for t_hull in t_hulls1]

        t_hulls2 = acd(t_m2)
        k_hulls2 = [get_kaolinmesh_from_trimesh(t_hull) for t_hull in t_hulls2]

        a_k_hulls1 = augment(k_hulls1)
        a_k_hulls2 = augment(k_hulls2)

        a_t_hulls1 = [get_trimesh_from_kaolinmesh(a_k_hull) for a_k_hull in a_k_hulls1]
        a_t_hulls2 = [get_trimesh_from_kaolinmesh(a_k_hull) for a_k_hull in a_k_hulls2]

        k_result = merge_meshes(a_t_hulls1 + a_t_hulls2)

        mesh, uv, texture = approximate_convex_decomposition(k_result, hull_num=8)

        for j in range(20):
            now_mesh = TriangleMesh.from_tensors(mesh.vertices.clone(), mesh.faces.clone())

            dist = 3.0 + torch.rand(1).item() * 2
            elev = (torch.rand(1).item() - 0.5) * 90
            azim = torch.rand(1).item() * 360

            rgb, silhouette, _ = PhongRenderer.render(now_mesh, dist, elev, azim, uv, texture)

            rgb = rgb[0].cpu().permute(2, 0, 1)
            silhouette = silhouette[0].cpu().permute(2, 0, 1)
            img = torch.cat([rgb, silhouette], 0)

            pil_img = to_pil(img)
            now_mesh.vertices = ShapeNetDataset.transform_to_view_center(now_mesh.vertices,
                                                                         dist=dist, elev=elev, azim=azim)

            pil_img.save(os.path.join(dataset_path, 'img_%.6d.png' % n))
            now_mesh.save_mesh(os.path.join(dataset_path, 'mesh_%.6d.obj' % n))
            json_data = json.dumps({'dist': dist, 'elev': elev, 'azim': azim})
            with open(os.path.join(dataset_path, 'meta_%.6d.json' % n), 'w') as f:
                f.write(json_data)
                f.close()
            n += 1


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    dataset_path = os.path.join(args.path, args.type)
    os.makedirs(dataset_path, exist_ok=True)

    if args.dataset == 'point_mixup':
        generate_point_mixup_dataset(dataset_path)
    elif args.dataset == 'acd':
        generate_acd_dataset(dataset_path, args.type)
    elif args.dataset == 'acd_mix':
        generate_acd_mix_dataset(dataset_path, args.type)


