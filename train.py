import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from modules import VPNet, ShapeNetDataset, Sampling, ChamferDistanceLoss, Meshing, Visualizer
from config import *


random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM

print('Load dataset...')
train_dataset = ShapeNetDataset('train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print('Dataset size =', len(train_dataset))

model = VPNet().to(DEVICE)
optimizer = Adam(params=model.parameters(), lr=LR)
cd_loss_func = ChamferDistanceLoss()
vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

classes_str = ''
for c in TRAIN_CLASSES:
    classes_str += c + '_'

dir_path = os.path.join(EXPERIMENT_PATH, 'train', classes_str)
checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')
os.makedirs(dir_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

for epoch_now in range(EPOCH_NUM):
    model.train()
    epoch_loss, n = 0.0, 0

    progress_bar = tqdm(train_dataloader)

    for data in progress_bar:
        imgs, points = data[0].to(DEVICE), data[1].to(DEVICE)

        volumes, rotates, translates = model(imgs)
        predict_points = []

        for i in range(vp_num):
            sampling = Sampling.cuboid_sampling if i < CUBOID_NUM else Sampling.sphere_sampling
            predict_points.append(sampling(volumes[i], rotates[i], translates[i], SAMPLE_NUM))

        predict_points = torch.cat(predict_points, dim=1)
        cd_loss = cd_loss_func(predict_points, points, CD_W1, CD_W2)

        optimizer.zero_grad()
        cd_loss.backward()
        optimizer.step()

        n += 1
        epoch_loss += cd_loss.item()
        progress_bar.set_description('CD Loss = %.6f' % cd_loss.item())

    print('Epoch %d, avg loss = %.6f' % (epoch_now + 1, epoch_loss / n))

    # Record some result
    if (epoch_now + 1) % 5 == 0:
        model.eval()

        volumes, rotates, translates = model(imgs)
        batch_vp_meshes = [[] for i in range(BATCH_SIZE)]

        for i in range(vp_num):
            meshing = Meshing.cuboid_meshing if i < CUBOID_NUM else Meshing.sphere_meshing

            for b in range(BATCH_SIZE):
                batch_vp_meshes[b].append(meshing(volumes[i], rotates[i], translates[i])[b])

        for b in range(BATCH_SIZE):
            img = imgs[b]
            vp_meshes = batch_vp_meshes[b]
            Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, 'epoch%d-%d.png' % (epoch_now+1, b)))

        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_epoch%03d.pth' % (epoch_now + 1)))
