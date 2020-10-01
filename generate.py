import os
from modules import ShapeNetDataset
from config import *


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NUM
    print('Load dataset...')
    train_dataset = ShapeNetDataset('train')
    train_dataset.save_view_center_dataset()
