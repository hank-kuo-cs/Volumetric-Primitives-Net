import os
import random
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from modules.network import VPNetOneRes, VPNetTwoRes
from modules.dataset import ShapeNetDataset
from modules.meshing import Meshing
from modules.sampling import Sampling
from modules.loss import ChamferDistanceLoss, SilhouetteLoss
from modules.visualize import Visualizer, TensorboardWriter
from modules.transform import view_to_obj_points
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
    parser.add_argument('-pre', '--pretrain_model', type=str,
                        help='Load a pretrained model to continue training')

    return parser.parse_args()


def set_file_path():
    classes_str = ''
    for c in TRAIN_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'train', classes_str)
    checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    return dir_path, checkpoint_path


def load_dataset():
    print('Load dataset...')
    train_dataset = ShapeNetDataset('train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print('Dataset size =', len(train_dataset))

    return train_dataloader


def load_model(pretrain_model_path: str):
    model = VPNetOneRes() if BACKBONE == 'vpnet_oneres' else VPNetTwoRes()
    model = model.to(DEVICE)

    if pretrain_model_path:
        print('Load pretrianed model:', pretrain_model_path)
        model.load_state_dict(torch.load(pretrain_model_path))

    if IS_FIX_VOLUME:
        print('Fix volume weight...')
        model.fix_volume_weight()

    return model


def load_optimizer(model):
    if not IS_DECAY_VOLUME_RES:
        return Adam(params=model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=W_DECAY)

    print('Optimizer use different lr...')

    if BACKBONE == 'vpnet_oneres':
        return Adam(params=[
            {'params': model.rotate_fc.parameters(), 'lr': LR},
            {'params': model.translate_fc.parameters(), 'lr': LR},
            {'params': model.resnet.parameters(), 'lr': LR * DECAY_VOLUME_RES_RATE},
            {'params': model.volume_fc.parameters(), 'lr': LR * DECAY_VOLUME_RES_RATE},
        ], betas=(0.9, 0.99), weight_decay=W_DECAY)

    elif BACKBONE == 'vpnet_twores':
        return Adam(params=[
            {'params': model.transform_resnet.parameters(), 'lr': LR},
            {'params': model.rotate_fc.parameters(), 'lr': LR},
            {'params': model.translate_fc.parameters(), 'lr': LR},
            {'params': model.volume_resnet.parameters(), 'lr': LR * DECAY_VOLUME_RES_RATE},
            {'params': model.volume_fc.parameters(), 'lr': LR * DECAY_VOLUME_RES_RATE},
        ], lr=LR, betas=(0.9, 0.99), weight_decay=W_DECAY)


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


def compose_vp_meshes(batch_vp_meshes):
    batch_meshes = []
    for b in range(len(batch_vp_meshes)):
        mesh = Meshing.compose_meshes(batch_vp_meshes[b])
        batch_meshes.append(mesh)

    return batch_meshes


def calculate_points_loss(predict_points, canonical_points, view_center_points, dists, elevs, azims):
    cd_loss_func = ChamferDistanceLoss()

    if not IS_VIEW_CENTER:
        return torch.tensor(0.0).to(DEVICE), cd_loss_func(predict_points, canonical_points) * L_CAN_CD

    predict_canonical_points = view_to_obj_points(predict_points, dists, elevs, azims)

    view_center_cd_loss = cd_loss_func(predict_points, view_center_points) * L_VIEW_CD
    obj_center_cd_loss = cd_loss_func(predict_canonical_points, canonical_points) * L_CAN_CD

    return view_center_cd_loss, obj_center_cd_loss


def calculate_silhouette_loss(predict_meshes, silhouettes, dists, elevs, azims):
    if L_SIL == 0:
        return torch.tensor(0.0).to(DEVICE)

    silhouette_loss_func = SilhouetteLoss()

    if IS_VIEW_CENTER:
        dists = torch.full_like(dists, fill_value=1.0).to(DEVICE)
        elevs, azims = torch.zeros_like(elevs).to(DEVICE), torch.zeros_like(azims).to(DEVICE)

    return silhouette_loss_func(predict_meshes, silhouettes, dists, elevs, azims) * L_SIL


def show_loss_one_tensorboard(epoch, avg_losses):
    if L_VIEW_CD:
        tensorboard_writer.add_scalar('train/view_cd_loss', epoch, avg_losses['view_cd'])
    if L_CAN_CD:
        tensorboard_writer.add_scalar('train/obj_cd_loss', epoch, avg_losses['obj_cd'])
    if L_SIL:
        tensorboard_writer.add_scalar('train/silhouette_loss', epoch, avg_losses['sil'])


def train(args):
    train_dataloader = load_dataset()
    dir_path, checkpoint_path = set_file_path()

    model = load_model(args.pretrain_model)
    optimizer = load_optimizer(model)

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        model.train()
        avg_losses = {'view_cd': 0.0, 'obj_cd': 0.0, 'sil': 0.0}
        n = 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
            canonical_points, view_center_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)
            dists, elevs, azims = data['dist'].float().to(DEVICE), data['elev'].float().to(DEVICE), data['azim'].float().to(DEVICE)

            volumes, rotates, translates = model(rgbs)

            # Chamfer Distance Loss
            predict_points = sample_predict_points(volumes, rotates, translates)

            view_cd_loss, obj_cd_loss = calculate_points_loss(predict_points, canonical_points, view_center_points,
                                                              dists, elevs, azims)

            # Silhouette Loss
            batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)
            predict_meshes = compose_vp_meshes(batch_vp_meshes)

            sil_loss = calculate_silhouette_loss(predict_meshes, silhouettes, dists, elevs, azims)

            total_loss = view_cd_loss + obj_cd_loss + sil_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            avg_losses['view_cd'] += view_cd_loss.item()
            avg_losses['obj_cd'] += obj_cd_loss.item()
            avg_losses['sil'] += sil_loss.item()
            progress_bar.set_description('View CD Loss = %.6f, Obj CD Loss = %.6f, Sil Loss = %.6f'
                                         % (view_cd_loss.item(), obj_cd_loss.item(), sil_loss.item()))

        avg_losses['view_cd'] /= n
        avg_losses['obj_cd'] /= n
        avg_losses['sil'] /= n

        print('Epoch %d. View CD Loss = %.6f. Obj CD Loss = %.6f, Sil Loss = %.6f\n'
              % (epoch_now + 1, avg_losses['view_cd'], avg_losses['obj_cd'], avg_losses['sil']))
        show_loss_one_tensorboard(epoch_now + 1, avg_losses)

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now + 1, b))
                Visualizer.render_vp_meshes(img, vp_meshes, save_name, dist=SHOW_DIST)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


if __name__ == '__main__':
    args = parse_args()
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    train(args)
