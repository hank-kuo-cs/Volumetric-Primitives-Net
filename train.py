import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from modules import VPNet, ShapeNetDataset, Sampling, ChamferDistanceLoss, Meshing, Visualizer, SilhouetteLoss
from config import *


def set_seed():
    manual_seed = 1234
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
    meshing_funcs = [Meshing.cuboid_meshing, Meshing.sphere_meshing, Meshing.]

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


def train():
    train_dataloader = load_dataset()
    dir_path, checkpoint_path = set_file_path()

    model = VPNet().to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=LR, betas=(0.9, 0.99))

    cd_loss_func = ChamferDistanceLoss()
    silhouette_loss_func = SilhouetteLoss()

    # Training Process
    for epoch_now in range(EPOCH_NUM):
        model.train()
        epoch_avg_cd_loss, n = 0.0, 0

        progress_bar = tqdm(train_dataloader)

        for data in progress_bar:
            rgbs, silhouettes, points = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
            dists, elevs, azims = data[3].to(DEVICE), data[4].to(DEVICE), data[5].to(DEVICE)

            volumes, rotates, translates = model(rgbs)

            # Chamfer Distance Loss
            predict_points = sample_predict_points(volumes, rotates, translates)
            cd_loss = cd_loss_func(predict_points, points) * L_CD

            # Silhouette Loss
            batch_vp_meshes = get_vp_meshes(volumes, rotates, translates)
            predict_meshes = compose_vp_meshes(batch_vp_meshes)

            sil_loss = silhouette_loss_func(predict_meshes, silhouettes, dists, elevs, azims) * L_SIL

            total_loss = cd_loss + sil_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n += 1
            epoch_avg_cd_loss += cd_loss.item()
            progress_bar.set_description('CD Loss = %.6f, Sil Loss = %.6f' % (cd_loss.item(), sil_loss.item()))

        print('Epoch %d, avg loss = %.6f\n' % (epoch_now + 1, epoch_avg_cd_loss / n))

        # Record some result
        if (epoch_now + 1) % 5 == 0:
            for b in range(BATCH_SIZE):
                img = rgbs[b]
                vp_meshes = batch_vp_meshes[b]
                Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now+1, b)))

            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))


if __name__ == '__main__':
    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    train()
