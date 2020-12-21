import os
import sys
import random
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.network import VPNetOneRes, VPNetTwoRes, DepthEstimationNet
from modules.sampling import Sampling
from modules.loss import ChamferDistanceLoss
from modules.meshing import Meshing
from modules.visualize import Visualizer
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


def load_model(pretrain_vpn_path, pretrain_den_path):
    vpn = VPNetOneRes() if BACKBONE == 'vpnet_oneres' else VPNetTwoRes()
    vpn = vpn.to(DEVICE)
    vpn.load_state_dict(torch.load(pretrain_vpn_path))

    den = DepthEstimationNet()
    den = den.to(DEVICE)
    den.load_state_dict(torch.load(pretrain_den_path))

    return vpn, den


def get_model_path(epoch):
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    depth_checkpoint_path = os.path.join(checkpoint_path, 'depth')

    vpn_path = os.path.join(checkpoint_path, 'model_epoch%03d.pth' % epoch)
    den_path = os.path.join(depth_checkpoint_path, 'model_epoch%03d.pth' % epoch)

    return vpn_path, den_path


def set_save_path(is_unseen=False):
    classes_str = ''
    classes = TEST_UNSEEN_CLASSES if is_unseen else TEST_SEEN_CLASSES
    for c in classes:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'test', classes_str)
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'depth'), exist_ok=True)

    return dir_path


@torch.no_grad()
def test(args):
    epoch = args.epoch
    is_unseen = args.is_unseen
    dir_path = set_save_path(is_unseen)
    test_dataloader = load_dataset(is_unseen)

    vpn_path, den_path = get_model_path(epoch)
    vpn, den = load_model(vpn_path, den_path)

    cd_loss_func = ChamferDistanceLoss()
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

    vpn.eval()
    den.eval()

    progress_bar = tqdm(test_dataloader)
    avg_cd_loss, n = 0.0, 0
    class_avg_cd_losses = [0.0 for i in range(len(class_names))]
    class_ns = [0 for i in range(len(class_names))]

    for data in progress_bar:
        rgbs, silhouettes, depths = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE), data['depth'].to(DEVICE)
        canonical_points, view_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)
        class_indices = data['class_index']
        # dists, elevs, azims = data['dist'].float().to(DEVICE), data['elev'].float().to(DEVICE), data['azim'].float().to(DEVICE)

        predict_depths = den(rgbs)
        volumes, rotates, translates, local_features, global_features = vpn(predict_depths)
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
        depth = depths[0]
        predict_depth = predict_depths[0]
        vp_meshes = batch_vp_meshes[0]

        Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, 'epoch%d-%d.png' % (epoch, n // 3)), SHOW_DIST)
        Visualizer.save_depth_imgs(predict_depth, depth, os.path.join(dir_path, 'depth', 'epoch%d-%d.png' % (epoch, n // 3)))

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
