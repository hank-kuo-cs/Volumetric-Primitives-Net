import os
import random
import argparse
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from modules.network import GCNModel, VPNetOneRes
from modules.loss import ChamferDistanceLoss, EarthMoverDistanceLoss
from modules.dataset import ACDMixDataset
from modules.sampling import Sampling
from modules.meshing import Meshing
from modules.visualize import Visualizer


TRAIN_CLASSES = ['airplane', 'car', 'chair']


def set_seed(manual_seed=1234):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vpn', type=str, help='Load a VP network to get initial vp prediction')
    parser.add_argument('-lr', type=float, default=1e-5, help='Learning rate of optimizer')
    parser.add_argument('-w_decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-gpu', type=str, default='0', help='GPU device number')
    parser.add_argument('-epochs', type=int, default=50, help='Epoch num')
    parser.add_argument('-batch', type=int, default=8, help='Batch size')
    parser.add_argument('-a_mix', '--acd_mix', type=str, help='Use acd mix dataset to train')
    parser.add_argument('-exp_name', type=str, help='Experiment name')

    return parser.parse_args()


def set_file_path(args):
    classes_str = ''
    for c in TRAIN_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join('experiment', 'train', classes_str)
    checkpoint_path = os.path.join('experiment', 'checkpoint')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    return dir_path, checkpoint_path


def sample_predict_points(volumes, rotates, translates):
    vp_num = 16
    sampling = Sampling.sphere_sampling
    predict_points = []

    for i in range(vp_num):
        predict_points.append(sampling(volumes[i], rotates[i], translates[i], 128))

    predict_points = torch.cat(predict_points, dim=1)
    return predict_points


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


def calculate_emd_loss(predict_points, gt_points):
    emd_loss_func = EarthMoverDistanceLoss()
    dist, assignment = emd_loss_func(predict_points, gt_points, 0.005, 50)

    return torch.sqrt(dist).mean()


def train(args):
    print('Load VPN: %s' % args.vpn)
    vpn = VPNetOneRes().cuda()
    vpn.load_state_dict(torch.load(args.vpn))
    vpn.eval()

    gcn = GCNModel().cuda()
    optimizer = Adam(params=gcn.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.w_decay)

    cd_loss_func = ChamferDistanceLoss()

    print('Load ACD Dataset: %s' % args.acd_mix)
    dataset = ACDMixDataset(args.acd_mix)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, num_workers=16)
    dir_path, checkpoint_path = set_file_path(args)

    for epoch in range(args.epochs):
        gcn.train()
        avg_losses = {'cd': 0.0, 'emd': 0.0}
        n = 0

        progress_bar = tqdm(dataloader)

        for data in progress_bar:
            rgbs, points, angles = data['rgb'].cuda(), data['points'].cuda(), data['rotate_angle'].cuda()
            # points = rotate_points_forward_x_axis(points, angles)

            volumes, rotates, translates, perceptual_features, global_features = vpn(rgbs)

            batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)
            predict_meshes = compose_vp_meshes(batch_vp_meshes)
            predict_vertices = gcn(predict_meshes, rgbs, perceptual_features, global_features)

            cd_loss = cd_loss_func(predict_vertices, points)
            emd_loss = calculate_emd_loss(predict_vertices, points)

            total_loss = cd_loss + emd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            avg_losses['cd'] += cd_loss.item()
            avg_losses['emd'] += emd_loss.item()

            progress_bar.set_description('CD Loss = %.6f, EMD Loss = %.6f' % (cd_loss.item(), emd_loss.item()))

        avg_losses['cd'] /= n
        avg_losses['emd'] /= n
        print('[Epoch %d AVG Loss] CD Loss = %.6f, EMD Loss = %.6f\n'
              % (epoch + 1, avg_losses['cd'], avg_losses['emd']))

        if (epoch + 1) % 5 == 0:
            for b in range(args.batch):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                vertices = predict_vertices[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch + 1, b))
                Visualizer.render_refine_vp_meshes(img, vp_meshes, vertices, save_name)

            torch.save(gcn.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch + 1)))


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train(args)
