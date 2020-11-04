# CUDA
DEVICE = 'cuda'
DEVICE_NUM = '6'

# Hyper parameter
LR = 1e-4
W_DECAY = 1e-6
SAMPLE_NUM = 128
BATCH_SIZE = 8
EPOCH_NUM = 50
CD_W1 = 1.0
CD_W2 = 1.0
L_VIEW_CD = 1.0
L_CAN_CD = 1.0
L_SIL = 0.0
L_VP_DIV = 0.0

# Network
MANUAL_SEED = 1234
BACKBONE = 'vpnet_oneres'  # vpnet_twores, vpnet_oneres
VP_CLAMP_MIN = 0.01
VP_CLAMP_MAX = 0.8
IS_DROPOUT = False
IS_SIGMOID = True
VOLUME_RESTRICT = [8, 10, 10]
SILHOUETTE_LOSS_FUNC = 'L1'  # L1 or MSE
IS_FIX_VOLUME = False
IS_DECAY_VOLUME_RES = False
DECAY_VOLUME_RES_RATE = 0.1

# Volumetric Primitives
CUBOID_NUM = 8
SPHERE_NUM = 8
CONE_NUM = 0
VP_NUM = CUBOID_NUM + SPHERE_NUM + CONE_NUM

# Visualize
EXPERIMENT_PATH = 'experiment'
SHOW_DIST = 1
TENSORBOARD_PATH = '/home/hank/3d/Tensorboard'
EXPERIMENT_NAME = ''

# Dataset
DATASET_ROOT = '/eva_data/hdd1/hank'
IS_VIEW_CENTER = True
IS_NORMALIZE = False
IS_DIST_INVARIANT = False
LITTLE_NUM = {'train': 100, 'test': 20}
IMG_SIZE = 128
AUGMENT_3D = {'rotate': False, 'cutmix': False, 'scale': False, 'point_mixup': False}

# airplane, rifle, display, table, telephone, car, chair, bench, lamp, cabinet, loudspeaker, sofa, watercraft
TRAIN_CLASSES = ['airplane', 'car', 'chair']
TEST_CLASSES = ['airplane', 'car', 'chair']
# TEST_CLASSES = ['rifle', 'display', 'table', 'telephone',
# 'bench', 'lamp', 'cabinet', 'loudspeaker', 'sofa', 'watercraft']
