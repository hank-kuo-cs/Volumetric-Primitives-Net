import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import VPNet, ShapeNetDataset, Sampling, ChamferDistanceLoss, Meshing, Visualizer
from config import *


random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM

print('Load dataset...')
test_dataset = ShapeNetDataset('test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print('Dataset size =', len(test_dataset))

epoch = int(input('Use epoch = '))
checkpoint_path = os.path.join(EXPERIMENT_PATH, 'checkpoint')

model = VPNet().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(checkpoint_path, '/model_epoch%03d.pth' % epoch)))

cd_loss_func = ChamferDistanceLoss()
vp_num = CUBOID_NUM + SPHERE_NUM + CONE_NUM

model.eval()

progress_bar = tqdm(test_dataloader)
cd_loss, n = 0.0, 0

for data in progress_bar:
    imgs, points = data[0].to(DEVICE), data[1].to(DEVICE)

    volumes, rotates, translates = model(imgs)
    predict_points = []

    for i in range(vp_num):
        sampling = Sampling.cuboid_sampling if i < CUBOID_NUM else Sampling.sphere_sampling
        predict_points.append(sampling(volumes[i], rotates[i], translates[i], SAMPLE_NUM))

    predict_points = torch.cat(predict_points, dim=1)
    cd_loss += cd_loss_func(predict_points, points, CD_W1, CD_W2).item()
    n += 1

print('Epoch %d, avg cd loss = %.6f' % (epoch, cd_loss / n))

# Record some result
volumes, rotates, translates = model(imgs)
batch_vp_meshes = [[] for i in range(BATCH_SIZE)]

for i in range(vp_num):
    meshing = Meshing.cuboid_meshing if i < CUBOID_NUM else Meshing.sphere_meshing

    for b in range(BATCH_SIZE):
        batch_vp_meshes[b].append(meshing(volumes[i], rotates[i], translates[i])[b])

classes_str = ''
for c in TEST_CLASSES:
    classes_str += c + '_'

dir_path = os.path.join(EXPERIMENT_PATH, 'test', classes_str)
os.makedirs(dir_path, exist_ok=True)

for b in range(BATCH_SIZE):
    img = imgs[b]
    vp_meshes = batch_vp_meshes[b]
    Visualizer.render_vp_meshes(img, vp_meshes, os.path.join(dir_path, '/epoch%d-%d.png' % (epoch, b)))
