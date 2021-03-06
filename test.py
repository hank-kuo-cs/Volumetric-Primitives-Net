import os
import sys
import random
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import VPNetOneRes, VPNetTwoRes, Sampling, ChamferDistanceLoss, Meshing, Visualizer
from modules.dataset import Classes, ShapeNetDataset
from config import *


class_names = Classes.get_class_names()


def set_seed(manual_seed=MANUAL_SEED):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if DEVICE != 'cpu':
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--is_unseen', action='store_true', help='Test on unseen classes')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Use which model train on this number of epochs')

    return parser.parse_args()


def load_dataset(is_unseen: bool):
    print('Load dataset...')
    test_dataset = ShapeNetDataset('test_unseen' if is_unseen else 'test_seen')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    print('Dataset size =', len(test_dataset))

    return test_dataloader


def load_model(pretrain_model_path):
    model = VPNetOneRes() if BACKBONE == 'vpnet_oneres' else VPNetTwoRes()
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(pretrain_model_path))
    return model


def get_model_path(epoch):
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    return os.path.join(checkpoint_path, 'model_epoch%03d.pth' % epoch)


def set_save_path(is_unseen=False):
    classes_str = ''
    classes = TEST_UNSEEN_CLASSES if is_unseen else TEST_SEEN_CLASSES
    for c in classes:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'test', classes_str)
    os.makedirs(dir_path, exist_ok=True)

    return dir_path


@torch.no_grad()
def test(args):
    epoch = args.epoch
    is_unseen = args.is_unseen
    dir_path = set_save_path(is_unseen)
    test_dataloader = load_dataset(is_unseen)
    model_path = get_model_path(epoch)
    model = load_model(model_path)

    cd_loss_func = ChamferDistanceLoss()
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

    model.eval()

    progress_bar = tqdm(test_dataloader)
    avg_cd_loss, n = 0.0, 0
    class_avg_cd_losses = [0.0 for i in range(len(class_names))]
    class_ns = [0 for i in range(len(class_names))]

    for data in progress_bar:
        rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
        canonical_points, view_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)
        class_indices = data['class_index']
        # dists, elevs, azims = data['dist'].float().to(DEVICE), data['elev'].float().to(DEVICE), data['azim'].float().to(DEVICE)

        volumes, rotates, translates, features = model(rgbs)
        predict_points = []

        for i in range(vp_num):
            sampling = Sampling.cuboid_sampling if i < CUBOID_NUM else Sampling.sphere_sampling
            predict_points.append(sampling(volumes[i], rotates[i], translates[i], SAMPLE_NUM))

        predict_points = torch.cat(predict_points, dim=1)
        cd_loss = cd_loss_func(predict_points, view_points, each_batch=True) * L_VIEW_CD if IS_VIEW_CENTER else \
            cd_loss_func(predict_points, canonical_points, each_batch=True) * L_CAN_CD

        avg_cd_loss += cd_loss.mean().item()
        for b in range(BATCH_SIZE):
            class_avg_cd_losses[class_indices[b]] += cd_loss[b].item()
            class_ns[class_indices[b]] += 1
        n += 1

        # Record some Result
        if n % 3 > 0:
            continue

        batch_vp_meshes = [[] for i in range(BATCH_SIZE)]

        for i in range(vp_num):
            meshing = Meshing.cuboid_meshing if i < CUBOID_NUM else Meshing.sphere_meshing

            for b in range(BATCH_SIZE):
                batch_vp_meshes[b].append(meshing(volumes[i], rotates[i], translates[i])[b])

        img = rgbs[0]
        vp_meshes = batch_vp_meshes[0]
        Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, 'epoch%d-%d.png' % (epoch, n // 3)), SHOW_DIST)

    avg_cd_loss /= n
    print('\nEpoch %d\n============================' % epoch)

    for i in range(len(class_avg_cd_losses)):
        if class_ns[i] == 0:
            continue
        class_avg_cd_losses[i] /= class_ns[i]
        print(class_names[i], 'avg cd loss = %.6f' % class_avg_cd_losses[i])

    print('============================\ntotal avg cd loss = %.6f' % avg_cd_loss)


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    test(args)
