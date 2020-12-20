import os
import random
import torch
from tqdm import tqdm
from kaolin.rep import TriangleMesh
from torch.utils.data import DataLoader
from torch.optim import Adam
from modules.dataset import ShapeNetDataset
from modules.network import SDNet
from modules.loss import ChamferDistanceLoss, SilhouetteLoss
from modules.visualize import TensorboardWriter, Visualizer
from modules.transform import rotate_points_forward_x_axis
from modules.augmentation import cut_mix_data
from config import *


tensorboard_writer = TensorboardWriter()


def set_seed(manual_seed=MANUAL_SEED):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if DEVICE != 'cpu':
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


def set_file_path():
    classes_str = ''
    for c in TRAIN_CLASSES:
        classes_str += c + '_'

    dir_path = os.path.join(EXPERIMENT_PATH, 'train_sphere', classes_str)
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


def show_loss_one_tensorboard(epoch, cd_loss):
    tensorboard_writer.add_scalar('train/sphere_cd_loss', epoch, cd_loss)


def train():
    train_dataloader = load_dataset()
    dir_path, checkpoint_path = set_file_path()

    model = SDNet().to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=W_DECAY)

    cd_loss_func = ChamferDistanceLoss()
    silhouette_loss_func = SilhouetteLoss()

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        model.train()
        epoch_avg_cd_loss, n = 0.0, 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            rgbs, silhouettes = data['rgb'].to(DEVICE), data['silhouette'].to(DEVICE)
            canonical_points, view_center_points = data['canonical_points'].to(DEVICE), data['view_center_points'].to(DEVICE)
            rotate_angles = data['rotate_angle'].float().to(DEVICE)
            dists, elevs, azims = data['dist'].float().to(DEVICE), data['elev'].float().to(DEVICE), data['azim'].float().to(DEVICE)

            sphere_meshes = load_sphere_meshes()
            vertices_offset = model(rgbs)

            if AUGMENT_3D['rotate']:
                view_center_points = rotate_points_forward_x_axis(view_center_points, rotate_angles)
            if AUGMENT_3D['cutmix']:
                rgbs, silhouettes, view_center_points = cut_mix_data(rgbs, silhouettes, view_center_points)

            # Chamfer Distance Loss
            predict_meshes = deform_meshes(sphere_meshes, vertices_offset)
            predict_points = sample_points(predict_meshes)
            cd_loss = cd_loss_func(predict_points, view_center_points) * L_VIEW_CD if IS_VIEW_CENTER \
                else cd_loss_func(predict_points, canonical_points) * L_CAN_CD

            # Silhouette Loss
            if IS_VIEW_CENTER:
                dists = torch.full_like(dists, fill_value=1.0).to(DEVICE)
                elevs, azims = torch.zeros_like(elevs).to(DEVICE), torch.zeros_like(azims).to(DEVICE)
            sil_loss = silhouette_loss_func(predict_meshes, silhouettes, dists, elevs, azims) * L_SIL

            total_loss = cd_loss + sil_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            epoch_avg_cd_loss += cd_loss.item()
            progress_bar.set_description('CD Loss = %.6f, Sil Loss = %.6f' % (cd_loss.item(), sil_loss.item()))

        print('Epoch %d, avg loss = %.6f\n' % (epoch_now + 1, epoch_avg_cd_loss / n))
        show_loss_one_tensorboard(epoch_now + 1, epoch_avg_cd_loss / n)

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                predict_mesh = predict_meshes[b]
                save_name = os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now + 1, b))
                Visualizer.render_mesh_gif(img, predict_mesh, save_name, SHOW_DIST)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


if __name__ == '__main__':
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    train()
