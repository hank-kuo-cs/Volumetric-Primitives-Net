import os
import random
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss

from modules.network import VPNetOneRes, VPNetTwoRes, DepthEstimationNet
from modules.dataset import ShapeNetDataset, PointMixUpDataset, ACDMixDataset
from modules.meshing import Meshing
from modules.sampling import Sampling
from modules.loss import ChamferDistanceLoss, SilhouetteLoss, VPDiverseLoss, EarthMoverDistanceLoss
from modules.visualize import Visualizer, TensorboardWriter
from modules.transform import view_to_obj_points, rotate_points_forward_x_axis
from modules.augmentation import cut_mix_data, point_mixup_data
from modules.render import DepthRenderer
from config import *


tensorboard_writer = TensorboardWriter()


def set_seed(manual_seed=MANUAL_SEED):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if DEVICE != 'cpu':
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pre_vpn', '--pretrain_vpn', type=str, help='Load a pretrained model to continue training')
    parser.add_argument('-pre_den', '--pretrain_den', type=str, help='Load a pretrained model to continue training')
    parser.add_argument('-p_mix', '--point_mixup', type=str, help='Use point mixup dataset to train')
    parser.add_argument('-a_mix', '--acd_mix', type=str, help='Use acd mix dataset to train')

    return parser.parse_args()


def set_file_path():
    classes_str = ''
    for c in TRAIN_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'train', classes_str)
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    depth_checkpoint_path = os.path.join(checkpoint_path, 'depth')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(depth_checkpoint_path, exist_ok=True)

    return dir_path, checkpoint_path, depth_checkpoint_path


def load_dataset(point_mixup_dataset_name=''):
    print('Load dataset...')
    train_dataset = ShapeNetDataset('train') \
        if not point_mixup_dataset_name else PointMixUpDataset(point_mixup_dataset_name)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    print('Dataset size =', len(train_dataset))

    return train_dataloader


def load_model(pretrain_vpn_path: str, pretrain_den_path: str):
    vpn = VPNetOneRes() if BACKBONE == 'vpnet_oneres' else VPNetTwoRes()
    vpn = vpn.to(DEVICE)

    den = DepthEstimationNet()
    den = den.to(DEVICE)

    if pretrain_vpn_path:
        print('Load pretrained volumetric primitve model:', pretrain_vpn_path)
        vpn.load_state_dict(torch.load(pretrain_vpn_path))

    if pretrain_den_path:
        print('Load pretrained depth estimation model:')
        den.load_state_dict(torch.load(pretrain_den_path))

    if IS_FIX_VOLUME:
        print('Fix volume weight...')
        vpn.fix_volume_weight()

    return vpn, den


def load_optimizer(vpn, den):
    if not IS_DECAY_VOLUME_RES:
        return Adam(params=[
            {'params': vpn.parameters(), 'lr': LR_VPN},
            {'params': den.parameters(), 'lr': LR_DEN}
        ], betas=(0.5, 0.9))

    print('Optimizer use different lr...')

    if BACKBONE == 'vpnet_oneres':
        return Adam(params=[
            {'params': vpn.rotate_fc.parameters(), 'lr': LR_VPN},
            {'params': vpn.translate_fc.parameters(), 'lr': LR_VPN},
            {'params': vpn.resnet.parameters(), 'lr': LR_VPN * DECAY_VOLUME_RES_RATE},
            {'params': vpn.volume_fc.parameters(), 'lr': LR_VPN * DECAY_VOLUME_RES_RATE},
        ], betas=(0.9, 0.99), weight_decay=W_DECAY)

    elif BACKBONE == 'vpnet_twores':
        return Adam(params=[
            {'params': vpn.transform_resnet.parameters(), 'lr': LR_VPN},
            {'params': vpn.rotate_fc.parameters(), 'lr': LR_VPN},
            {'params': vpn.translate_fc.parameters(), 'lr': LR_VPN},
            {'params': vpn.volume_resnet.parameters(), 'lr': LR_VPN * DECAY_VOLUME_RES_RATE},
            {'params': vpn.volume_fc.parameters(), 'lr': LR_VPN * DECAY_VOLUME_RES_RATE},
        ], lr=LR_VPN, betas=(0.9, 0.99), weight_decay=W_DECAY)


def sample_predict_points(volumes, rotates, translates):
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM
    sampling_funcs = [Sampling.cuboid_sampling, Sampling.sphere_sampling, Sampling.cone_sampling]

    predict_points = []
    sample_type = 0

    for i in range(vp_num):
        if i == CUBOID_NUM or i == CUBOID_NUM + SPHERE_NUM:
            sample_type += 1

        sampling = sampling_funcs[sample_type]
        predict_points.append(sampling(volumes[i], rotates[i], translates[i], SAMPLE_NUM))

    predict_points = torch.cat(predict_points, dim=1)
    return predict_points


def get_vp_meshes(volumes, rotates, translates):
    vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM
    meshing_funcs = [Meshing.cuboid_meshing, Meshing.sphere_meshing, Meshing.cone_meshing]

    batch_vp_meshes = [[] for i in range(BATCH_SIZE)]  # (B, K)
    meshing_type = 0

    for i in range(vp_num):
        if i == CUBOID_NUM or i == CUBOID_NUM + SPHERE_NUM:
            meshing_type += 1

        meshing = meshing_funcs[meshing_type]
        meshes = meshing(volumes[i], rotates[i], translates[i])  # (B)

        for b in range(BATCH_SIZE):
            batch_vp_meshes[b].append(meshes[b])

    return batch_vp_meshes


def compose_vp_meshes(batch_vp_meshes) -> list:  # (N, K) -> (N)
    batch_meshes = []
    for b in range(len(batch_vp_meshes)):
        mesh = Meshing.compose_meshes(batch_vp_meshes[b])
        batch_meshes.append(mesh)

    return batch_meshes


def calculate_cd_loss(predict_points, canonical_points, view_center_points, dists, elevs, azims, angles):
    cd_loss_func = ChamferDistanceLoss()

    if not IS_VIEW_CENTER:
        return torch.tensor(0.0).to(DEVICE), cd_loss_func(predict_points, canonical_points) * L_CAN_CD

    predict_canonical_points = view_to_obj_points(predict_points, dists, elevs, azims, angles)

    view_center_cd_loss = cd_loss_func(predict_points, view_center_points) * L_VIEW_CD
    obj_center_cd_loss = cd_loss_func(predict_canonical_points, canonical_points) * L_CAN_CD

    return view_center_cd_loss, obj_center_cd_loss


def calculate_silhouette_loss(predict_meshes, silhouettes, dists, elevs, azims):
    if not L_SIL:
        return torch.tensor(0.0).to(DEVICE)

    silhouette_loss_func = SilhouetteLoss()

    if IS_VIEW_CENTER:
        dists = torch.full_like(dists, fill_value=1.0).to(DEVICE)
        elevs, azims = torch.zeros_like(elevs).to(DEVICE), torch.zeros_like(azims).to(DEVICE)

    return silhouette_loss_func(predict_meshes, silhouettes, dists, elevs, azims) * L_SIL


def calculate_vp_div_loss(translates, gt_points):
    if not L_VP_DIV:
        return torch.tensor(0.0).to(DEVICE)

    vp_div_loss_func = VPDiverseLoss()

    return vp_div_loss_func(translates, gt_points) * L_VP_DIV


def calculate_emd_loss(predict_points, gt_points):
    if not L_EMD:
        return torch.tensor(0.0).to(DEVICE)

    emd_loss_func = EarthMoverDistanceLoss()
    dist, assignment = emd_loss_func(predict_points, gt_points, 0.005, 50)

    return torch.sqrt(dist).mean()


def show_loss_on_tensorboard(epoch, avg_losses):
    if L_VIEW_CD:
        tensorboard_writer.add_scalar('train/view_cd_loss', epoch, avg_losses['view_cd'])
    if L_CAN_CD:
        tensorboard_writer.add_scalar('train/obj_cd_loss', epoch, avg_losses['obj_cd'])
    if L_SIL:
        tensorboard_writer.add_scalar('train/silhouette_loss', epoch, avg_losses['sil'])
    if L_VP_DIV:
        tensorboard_writer.add_scalar('train/vp_diverse_loss', epoch, avg_losses['vp_div'])
    if L_EMD:
        tensorboard_writer.add_scalar('train/emd_loss', epoch, avg_losses['emd'])
    if L_DEPTH:
        tensorboard_writer.add_scalar('train/depth_loss', epoch, avg_losses['depth'])


def train(args):
    train_dataset = ShapeNetDataset('train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    dir_path, checkpoint_path = set_file_path()

    model = load_model(args.pretrain_model)
    optimizer = load_optimizer(model)

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        model.train()
        avg_losses = {'view_cd': 0.0, 'obj_cd': 0.0, 'sil': 0.0, 'vp_div': 0.0, 'emd': 0.0}
        n = 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
            rotate_angles = data['rotate_angle'].float().to(DEVICE)
            canonical_points, view_center_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)
            dists, elevs, azims = data['dist'].float().to(DEVICE), data['elev'].float().to(DEVICE), data['azim'].float().to(DEVICE)

            if AUGMENT_3D['rotate']:
                view_center_points = rotate_points_forward_x_axis(view_center_points, rotate_angles)
            if AUGMENT_3D['cutmix']:
                rgbs, silhouettes, view_center_points = cut_mix_data(rgbs, silhouettes, view_center_points)
            if AUGMENT_3D['point_mixup']:
                rgbs, silhouettes, view_center_points = point_mixup_data(view_center_points)

            volumes, rotates, translates, features = model(rgbs)

            # Chamfer Distance Loss
            predict_points = sample_predict_points(volumes, rotates, translates)

            view_cd_loss, obj_cd_loss = calculate_cd_loss(predict_points, canonical_points, view_center_points,
                                                          dists, elevs, azims, rotate_angles)

            # Silhouette Loss
            batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)
            predict_meshes = compose_vp_meshes(batch_vp_meshes)

            sil_loss = calculate_silhouette_loss(predict_meshes, silhouettes, dists, elevs, azims)

            # VP Diverse Loss
            vp_div_loss = calculate_vp_div_loss(translates, view_center_points if IS_VIEW_CENTER else canonical_points)

            # EMD Loss
            emd_loss = calculate_emd_loss(predict_points, view_center_points)

            total_loss = view_cd_loss + obj_cd_loss + sil_loss + vp_div_loss + emd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            avg_losses['view_cd'] += view_cd_loss.item()
            avg_losses['obj_cd'] += obj_cd_loss.item()
            avg_losses['sil'] += sil_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            avg_losses['emd'] += emd_loss.item()

            progress_bar.set_description(
                'View CD Loss = %.6f, Obj CD Loss = %.6f, Sil Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f'
                % (view_cd_loss.item(), obj_cd_loss.item(), sil_loss.item(), vp_div_loss.item(), emd_loss.item()))

        avg_losses['view_cd'] /= n
        avg_losses['obj_cd'] /= n
        avg_losses['sil'] /= n
        avg_losses['vp_div'] /= n
        avg_losses['emd'] /= n

        print('[Epoch %d AVG Loss] View CD Loss = %.6f. Obj CD Loss = %.6f, Sil Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f\n'
              % (epoch_now + 1, avg_losses['view_cd'], avg_losses['obj_cd'], avg_losses['sil'], avg_losses['vp_div'], avg_losses['emd']))
        show_loss_on_tensorboard(epoch_now + 1, avg_losses)

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now + 1, b))
                Visualizer.render_vp_meshes(img, vp_meshes, save_name, dist=SHOW_DIST)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


def train_pointmixup(args):
    train_dataset = PointMixUpDataset(args.point_mixup)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    dir_path, checkpoint_path = set_file_path()

    model = load_model(args.pretrain_model)
    optimizer = load_optimizer(model)

    cd_loss_func = ChamferDistanceLoss()

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        model.train()
        avg_losses = {'view_cd': 0.0, 'vp_div': 0.0, 'emd': 0.0}
        n = 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
            points, angles = data['points'].to(DEVICE), data['angle'].float().to(DEVICE)

            if AUGMENT_3D['rotate']:
                points = rotate_points_forward_x_axis(points, angles)

            volumes, rotates, translates, features = model(rgbs)

            # Chamfer Distance Loss
            predict_points = sample_predict_points(volumes, rotates, translates)

            cd_loss = cd_loss_func(predict_points, points) * L_VIEW_CD
            vp_div_loss = calculate_vp_div_loss(translates, points) * L_VP_DIV
            emd_loss = calculate_emd_loss(predict_points, points) * L_EMD

            total_loss = cd_loss + vp_div_loss + emd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            avg_losses['view_cd'] += cd_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            avg_losses['emd'] += emd_loss.item()

            progress_bar.set_description('CD Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f'
                                         % (cd_loss.item(), vp_div_loss.item(), emd_loss.item()))

        avg_losses['view_cd'] /= n
        avg_losses['vp_div'] /= n
        avg_losses['emd'] /= n

        print('[Epoch %d AVG Loss] CD Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f\n'
              % (epoch_now + 1, avg_losses['view_cd'], avg_losses['vp_div'], avg_losses['emd']))
        show_loss_on_tensorboard(epoch_now + 1, avg_losses)

        batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now + 1, b))
                Visualizer.render_vp_meshes(img, vp_meshes, save_name, dist=SHOW_DIST)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


def train_acdmix(args):
    train_dataset = ACDMixDataset(args.acd_mix)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    dir_path, checkpoint_path, depth_checkpoint_path = set_file_path()

    vpn, den = load_model(args.pretrain_vpn, args.pretrain_den)
    optimizer = load_optimizer(vpn, den)

    cd_loss_func = ChamferDistanceLoss()
    l1_loss_func = L1Loss()

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        vpn.train()
        avg_losses = {'view_cd': 0.0, 'vp_div': 0.0, 'emd': 0.0, 'depth': 0.0}
        n = 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            # Load data
            rgbs, silhouettes, depths = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE), data['depth'].to(DEVICE)
            points, angles = data['points'].to(DEVICE), data['rotate_angle'].float().to(DEVICE)

            if AUGMENT_3D['rotate']:
                points = rotate_points_forward_x_axis(points, angles)

            # Predict
            predict_depths = den(rgbs)
            volumes, rotates, translates, features = vpn(predict_depths)

            # Loss
            predict_points = sample_predict_points(volumes, rotates, translates)

            cd_loss = cd_loss_func(predict_points, points) * L_VIEW_CD
            vp_div_loss = calculate_vp_div_loss(translates, points) * L_VP_DIV
            emd_loss = calculate_emd_loss(predict_points, points) * L_EMD

            depth_loss = l1_loss_func(predict_depths, depths) * L_DEPTH

            total_loss = cd_loss + vp_div_loss + emd_loss + depth_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            avg_losses['view_cd'] += cd_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            avg_losses['emd'] += emd_loss.item()
            avg_losses['depth'] += depth_loss.item()

            progress_bar.set_description('CD Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f, Depth Loss = %.6f'
                                         % (cd_loss.item(), vp_div_loss.item(), emd_loss.item(), depth_loss.item()))

        avg_losses['view_cd'] /= n
        avg_losses['vp_div'] /= n
        avg_losses['emd'] /= n
        avg_losses['depth'] /= n

        print('[Epoch %d AVG Loss] CD Loss = %.6f, VP Div Loss = %.6f, EMD Loss = %.6f, Depth Loss = %.6f\n'
              % (epoch_now + 1, avg_losses['view_cd'], avg_losses['vp_div'], avg_losses['emd'], avg_losses['depth']))
        show_loss_on_tensorboard(epoch_now + 1, avg_losses)

        batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now + 1, b))
                Visualizer.render_vp_meshes(img, vp_meshes, save_name, dist=SHOW_DIST)

            torch.save(vpn.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))
            torch.save(den.state_dict(), os.path.join(depth_checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    if args.point_mixup:
        train_pointmixup(args)
    elif args.acd_mix:
        train_acdmix(args)
    else:
        train(args)
