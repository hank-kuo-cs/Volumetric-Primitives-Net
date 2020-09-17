# CUDA
DEVICE = 'cuda'
DEVICE_NUM = '6'

# HYPER PARAMETER
LR = 1e-4
SAMPLE_NUM = 200
BATCH_SIZE = 4
EPOCH_NUM = 100
CD_W1 = 1.0
CD_W2 = 0.5
VP_CLAMP_MIN = 0.1
VP_CLAMP_MAX = 0.8

# PATH
EXPERIMENT_PATH = 'experiment/100_chairs/adam_batch4_lr1e-4_cd1and5e-1_sample200'
LITTLE_NUM = {'train': 100, 'test': 20}

# Volumetric Primitives
CUBOID_NUM = 6
SPHERE_NUM = 6
CONE_NUM = 0

# Dataset
IMG_SIZE = 128
# airplane, rifle, display, table, telephone, car, chair, bench, lamp, cabinet, loudspeaker, sofa, watercraft
TRAIN_CLASSES = ['chair']
TEST_CLASSES = ['chair']
DATASET_ROOT = '/eva_data/hank'
