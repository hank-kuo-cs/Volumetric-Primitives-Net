import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from modules import ShapeNetDataset
from modules.augmentation.point_mixup import generate_point_mixup_data
from modules.transform import rotate_points_forward_x_axis
from config import *


def load_dataset():
    print('Load dataset...')
    train_dataset = ShapeNetDataset('train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    print('Dataset size =', len(train_dataset))

    return train_dataloader


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    dataset_path = os.path.join(DATASET_ROOT, 'PointMixUp/1')
    os.makedirs(dataset_path, exist_ok=True)

    dataloader = load_dataset()
    progress_bar = tqdm(dataloader)

    to_pil = transforms.ToPILImage()
    n = 1

    for epoch in range(1):
        for data in progress_bar:
            rotate_angles = data['rotate_angle'].float().to(DEVICE)
            view_center_points = data['view_center_points'].to(DEVICE)

            # view_center_points = rotate_points_forward_x_axis(view_center_points, rotate_angles)
            rgbs, silhouettes, meshes = generate_point_mixup_data(view_center_points)

            for i in range(rgbs.size(0)):
                rgb = to_pil(rgbs[i].cpu())
                silhouette = to_pil(silhouettes[i].cpu())

                rgb.save(os.path.join(dataset_path, 'rgb_%d.png' % n))
                silhouette.save(os.path.join(dataset_path, 'silhouette_%d.png' % n))

                mesh = meshes[i]
                mesh.save_mesh(os.path.join(dataset_path, 'mesh_%d.obj' % n))

                n += 1
