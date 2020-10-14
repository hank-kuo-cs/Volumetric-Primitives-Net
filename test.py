import os
import sys
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import VPNet, ShapeNetDataset, Sampling, ChamferDistanceLoss, Meshing, Visualizer, VPNetOneRes
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


def load_model(pretrain_model_path):
    model = VPNet() if BACKBONE == 'vpnet' else VPNetOneRes()
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(pretrain_model_path))
    return model


def get_model_path(epoch):
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    return os.path.join(checkpoint_path, 'model_epoch%03d.pth' % epoch)


def test(epoch: int):
    test_dataloader = load_dataset()
    model_path = get_model_path(epoch)
    model = load_model(model_path)

    cd_loss_func = ChamferDistanceLoss()
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

    model.eval()

    progress_bar = tqdm(test_dataloader)
    avg_cd_loss, n = 0.0, 0

    for data in progress_bar:
        rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
        canonical_points, view_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)

        volumes, rotates, translates = model(rgbs)
        predict_points = []

        for i in range(vp_num):
            sampling = Sampling.cuboid_sampling if i < CUBOID_NUM else Sampling.sphere_sampling
            predict_points.append(sampling(volumes[i], rotates[i], translates[i], SAMPLE_NUM))

        predict_points = torch.cat(predict_points, dim=1)
        cd_loss = cd_loss_func(predict_points, view_points) * L_VIEW_CD if IS_VIEW_CENTER else \
            cd_loss_func(predict_points, canonical_points) * L_CAN_CD

        avg_cd_loss += cd_loss.item()
        n += 1

    avg_cd_loss /= n
    print('Epoch %d, avg cd loss = %.6f' % (epoch, avg_cd_loss))

    # Record some result
    volumes, rotates, translates = model(rgbs)
    batch_vp_meshes = [[] for i in range(BATCH_SIZE)]

    for i in range(vp_num):
        meshing = Meshing.cuboid_meshing if i < CUBOID_NUM else Meshing.sphere_meshing

        for b in range(BATCH_SIZE):
            batch_vp_meshes[b].append(meshing(volumes[i], rotates[i], translates[i])[b])

    classes_str = ''
    for c in TEST_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'test', classes_str)
    os.makedirs(dir_path, exist_ok=True)

    for b in range(BATCH_SIZE):
        img = rgbs[b]
        vp_meshes = batch_vp_meshes[b]
        Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, 'epoch%d-%d.png' % (epoch, b)), SHOW_DIST)


if __name__ == '__main__':
    epoch = int(sys.argv[1])
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    test(epoch)
