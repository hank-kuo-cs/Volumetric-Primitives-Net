# CUDA
DEVICE = 'cuda'
DEVICE_NUM = '5'

# Hyper parameter
LR = 1e-4
W_DECAY = 1e-6
SAMPLE_NUM = 200
BATCH_SIZE = 8
EPOCH_NUM = 50
CD_W1 = 1.0
CD_W2 = 1.0
L_CD = 1.0
L_SIL = 0.0

# Dataset
IS_NORMALIZE = False

# Network
MANUAL_SEED = 1234
BACKBONE = 'resnet18'  # resnet18, vgg19, resnet50
VP_CLAMP_MIN = 0.01
VP_CLAMP_MAX = 0.8
IS_DROPOUT = False
IS_SIGMOID = True
VOLUME_RESTRICT = [8, 10, 10]
SILHOUETTE_LOSS_FUNC = 'L1'  # L1 or MSE

# Path
EXPERIMENT_PATH = 'experiment'
LITTLE_NUM = {'train': 100, 'test': 20}

# Volumetric Primitives
CUBOID_NUM = 8
SPHERE_NUM = 8
CONE_NUM = 0

# Dataset
IS_VIEW_CENTER = True
IMG_SIZE = 128
# airplane, rifle, display, table, telephone, car, chair, bench, lamp, cabinet, loudspeaker, sofa, watercraft
TRAIN_CLASSES = ['chair']
TEST_CLASSES = ['chair']
DATASET_ROOT = '/eva_data/hank'
