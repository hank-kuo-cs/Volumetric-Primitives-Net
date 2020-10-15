import os
import sys
import random
import torch
from tqdm import tqdm
from kaolin.rep import TriangleMesh
from torch.utils.data import DataLoader
from modules import SDNet, ShapeNetDataset, ChamferDistanceLoss, Visualizer, SilhouetteLoss
from config import *


def set_seed(manual_seed=MANUAL_SEED):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if DEVICE != 'cpu':
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def load_dataset():
    print('Load dataset...')
    test_dataset = ShapeNetDataset('test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print('Dataset size =', len(test_dataset))

    return test_dataloader


def get_model_path(epoch):
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    return os.path.join(checkpoint_path, 'model_epoch%03d.pth' % epoch)


def load_one_sphere():
    obj_path = '386.obj'
    sphere_mesh = TriangleMesh.from_obj(obj_path)
    sphere_mesh.cuda()
    return sphere_mesh


def load_sphere_meshes():
    return [load_one_sphere() for b in range(BATCH_SIZE)]


def deform_meshes(meshes: list, vertices_offset: torch.Tensor):
    assert len(meshes) == vertices_offset.size(0) == BATCH_SIZE

    for b in range(BATCH_SIZE):
        meshes[b].vertices += vertices_offset[b]

    return meshes


def sample_points(meshes: list):
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM
    points = []

    for b in range(BATCH_SIZE):
        points.append(meshes[b].sample(SAMPLE_NUM * vp_num)[0][None])

    points = torch.cat(points, dim=0)

    return points


def test(epoch: int):
    test_dataloader = load_dataset()
    model_path = get_model_path(epoch)

    model = SDNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    cd_loss_func = ChamferDistanceLoss()
    model.eval()

    progress_bar = tqdm(test_dataloader)
    avg_cd_loss, n = 0.0, 0

    for data in progress_bar:
        rgbs, silhouettes, points = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE), data['points'].to(DEVICE)
        dists, elevs, azims = data['dist'].to(DEVICE), data['elev'].to(DEVICE), data['azim'].to(DEVICE)

        sphere_meshes = load_sphere_meshes()
        vertices_offset = model(rgbs)

        # Chamfer Distance Loss
        predict_meshes = deform_meshes(sphere_meshes, vertices_offset)
        predict_points = sample_points(predict_meshes)
        cd_loss = cd_loss_func(predict_points, points) * L_VIEW_CD

        avg_cd_loss += cd_loss.item()
        n += 1

    avg_cd_loss /= n
    print('Epoch %d, avg cd loss = %.6f' % (epoch, avg_cd_loss))

    # Record some result
    classes_str = ''
    for c in TEST_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'test_sphere', classes_str)
    os.makedirs(dir_path, exist_ok=True)

    for b in range(BATCH_SIZE):
        img, mesh = rgbs[b], predict_meshes[b]
        dist, elev, azim = dists[b].item(), elevs[b].item(), azims[b].item()

        save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch, b))
        Visualizer.render_mesh_3pose(img, mesh, save_name, dist, elev, azim)


if __name__ == '__main__':
    epoch = int(sys.argv[1])
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    test(epoch)
