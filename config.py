# CUDA
DEVICE = 'cuda'
DEVICE_NUM = ''

# Hyper parameter
LR = 1e-4
W_DECAY = 1e-6
SAMPLE_NUM = 128
BATCH_SIZE = 8
EPOCH_NUM = 50
CD_W1 = 1.0
CD_W2 = 1.0
L_VIEW_CD = 1.0
L_CAN_CD = 0.0
L_SIL = 0.0
L_VP_DIV = 0.1
L_EMD = 1.0

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
CUBOID_NUM = 0
SPHERE_NUM = 16
CONE_NUM = 0
VP_NUM = CUBOID_NUM + SPHERE_NUM + CONE_NUM

# Visualize
EXPERIMENT_PATH = 'experiment'
SHOW_DIST = 1
TENSORBOARD_PATH = '/home/hank/3d/Tensorboard'
EXPERIMENT_NAME = ''

# Dataset
DATASET_ROOT = '/eva_data/hdd1/hank'
GENRE_TESTING_ROOT = '/eva_data/hdd1/hank/GenRe/test'
LITTLE_NUM = {'train': 1000, 'test': 200}
AUGMENT_3D = {'rotate': False, 'cutmix': False, 'scale': False, 'point_mixup': False}
IMG_SIZE = 128
IS_VIEW_CENTER = True
IS_NORMALIZE = False
IS_DIST_INVARIANT = False
DECOMPOSE_CONVEX_NUM = 16

# airplane, rifle, display, table, telephone, car, chair, bench, lamp, cabinet, loudspeaker, sofa, watercraft
TRAIN_CLASSES = ['airplane', 'car', 'chair']
TEST_SEEN_CLASSES = ['airplane', 'car', 'chair']
TEST_UNSEEN_CLASSES = ['rifle', 'display', 'table', 'telephone',
                       'bench', 'lamp', 'cabinet', 'loudspeaker', 'sofa', 'watercraft']
