import os
import sys
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import Classes, ShapeNetDataset, GenReDataset
from modules.network import VPNetOneRes, GCNModel
from modules.loss import ChamferDistanceLoss, EarthMoverDistanceLoss
from modules.meshing import Meshing
from modules.visualize import Visualizer


class_names = Classes.get_class_names()
TEST_SEEN_CLASSES = ['airplane', 'car', 'chair']
TEST_UNSEEN_CLASSES = ['rifle', 'display', 'table', 'telephone', 'bench',
                       'lamp', 'cabinet', 'loudspeaker', 'sofa', 'watercraft']


def set_seed(manual_seed=1234):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--is_unseen', action='store_true', help='Test on unseen classes')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Use which model train on this number of epochs')
    parser.add_argument('-vpn', type=str, help='Load a VP network to get initial vp prediction')
    parser.add_argument('-batch', type=int, default=8, help='Batch size')
    parser.add_argument('-genre', action='store_true', help='Use GenRe testing dataset to test')

    return parser.parse_args()


def load_dataset(args):
    print('Load dataset...')
    test_dataset = ShapeNetDataset('test_unseen' if args.is_unseen else 'test_seen') \
        if not args.genre else GenReDataset()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False, num_workers=16)
    print('Dataset size =', len(test_dataset))

    return test_dataloader


def load_model(args):
    vpn = VPNetOneRes().cuda()
    vpn.load_state_dict(torch.load(args.vpn))

    gcn = GCNModel().cuda()
    gcn.load_state_dict(torch.load(get_model_path(args.epoch)))

    return vpn, gcn


def get_model_path(epoch):
    checkpoint_path = os.path.join('experiment', 'checkpoint')
    return os.path.join(checkpoint_path, 'model_epoch%03d.pth' % epoch)


def set_save_path(args):
    classes_str = ''
    classes = TEST_UNSEEN_CLASSES if args.is_unseen else TEST_SEEN_CLASSES
    for c in classes:
        classes_str += c + '_'

    dir_path = os.path.join('experiment', 'test', classes_str)
    os.makedirs(dir_path, exist_ok=True)

    return dir_path


def calculate_emd_loss(predict_points, gt_points):
    emd_loss_func = EarthMoverDistanceLoss()
    dist, assignment = emd_loss_func(predict_points, gt_points, 0.005, 50)

    return torch.sqrt(dist).mean(1)  # (B)


def calculate_cd_loss(predict_points, gt_points):
    cd_loss_func = ChamferDistanceLoss()

    return cd_loss_func(predict_points, gt_points, each_batch=True)  # (B)


def get_vp_meshes(volumes, rotates, translates):
    vp_num = 16
    meshing = Meshing.sphere_meshing

    batch_vp_meshes = [[] for i in range(volumes[0].size(0))]  # (B, K)

    for i in range(vp_num):
        meshes = meshing(volumes[i], rotates[i], translates[i])  # (B)

        for b in range(volumes[0].size(0)):
            batch_vp_meshes[b].append(meshes[b])
    return batch_vp_meshes


def compose_vp_meshes(batch_vp_meshes) -> list:  # (B, K) -> (B)
    batch_meshes = []
    for b in range(len(batch_vp_meshes)):
        mesh = Meshing.compose_meshes(batch_vp_meshes[b])
        batch_meshes.append(mesh)

    return batch_meshes


@torch.no_grad()
def test(args):
    dir_path = set_save_path(args)
    dataloader = load_dataset(args)
    vpn, gcn = load_model(args)

    vpn.eval()
    gcn.eval()

    progress_bar = tqdm(dataloader)

    n = 0
    avg_losses = {'cd': 0.0, 'emd': 0.0}
    class_avg_losses = {'cd': [0.0 for i in range(len(class_names))], 'emd': [0.0 for i in range(len(class_names))]}
    class_n = [0 for i in range(len(class_names))]

    for data in progress_bar:
        rgbs = data['rgb'].cuda()
        gt_points = data['view_center_points'].cuda()
        class_indices = data['class_index']

        volumes, rotates, translates, perceptual_features, global_features = vpn(rgbs)

        batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)
        predict_meshes = compose_vp_meshes(batch_vp_meshes)
        predict_vertices = gcn(predict_meshes, rgbs, perceptual_features, global_features)

        batch_cd_loss = calculate_cd_loss(predict_vertices, gt_points)
        batch_emd_loss = calculate_emd_loss(predict_vertices, gt_points)

        avg_losses['cd'] += batch_cd_loss.mean().item()
        avg_losses['emd'] += batch_emd_loss.mean().item()
        n += 1

        for b in range(len(batch_cd_loss)):
            class_avg_losses['cd'][class_indices[b]] += batch_cd_loss[b].item()
            class_avg_losses['emd'][class_indices[b]] += batch_emd_loss[b].item()
            class_n[class_indices[b]] += 1

        if n % 10 > 0:
            continue

        img = rgbs[0]
        vp_meshes = batch_vp_meshes[0]
        vertices = predict_vertices[0]
        save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (args.epoch, n // 10))
        Visualizer.render_refine_vp_meshes(img, vp_meshes, vertices, save_name)

    avg_losses['cd'] /= n
    avg_losses['emd'] /= n
    print('\nEpoch %d\n' % args.epoch)
    print('=' * 30)

    for i in range(len(class_n)):
        if class_n[i] == 0:
            continue
        class_avg_losses['cd'][i] /= class_n[i]
        class_avg_losses['emd'][i] /= class_n[i]

        print(class_names[i], '\t\tcd loss = %.6f, emd loss = %.6f' %
              (class_avg_losses['cd'][i], class_avg_losses['emd'][i]))
    print('=' * 30)
    print('total \t\tcd loss = %.6f, emd loss = %.6f'
          % (avg_losses['cd'], avg_losses['emd']))


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    test(args)
