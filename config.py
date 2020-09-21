# CUDA
DEVICE = 'cuda'
DEVICE_NUM = '6'

# Hyper parameter
LR = 1e-4
SAMPLE_NUM = 200
BATCH_SIZE = 8
EPOCH_NUM = 50
CD_W1 = 1.0
CD_W2 = 0.5
L_CD = 1.0
L_SIL = 0.01

# Dataset
IS_NORMALIZE = False

# Network
MANUAL_SEED = 1234
BACKBONE = 'resnet18'  # resnet18, vgg19, resnet50
VP_CLAMP_MIN = 0.0
VP_CLAMP_MAX = 1.0
IS_DROPOUT = False
IS_DEPTH_RESTRICT = False
SILHOUETTE_LOSS_FUNC = 'L1'

# Path
EXPERIMENT_PATH = 'experiment/100_chairs'
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
