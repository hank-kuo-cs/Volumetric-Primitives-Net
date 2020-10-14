# CUDA
DEVICE = 'cuda'
DEVICE_NUM = '6'

# Hyper parameter
LR = 1e-4
W_DECAY = 1e-6
SAMPLE_NUM = 200
BATCH_SIZE = 8
EPOCH_NUM = 50
CD_W1 = 1.0
CD_W2 = 1.0
L_VIEW_CD = 1.0
L_CAN_CD = 1.0
L_SIL = 0.0

# Network
MANUAL_SEED = 1234
BACKBONE = 'vpnet'  # vpnet, vpnet_oneres
VP_CLAMP_MIN = 0.01
VP_CLAMP_MAX = 0.8
IS_DROPOUT = False
IS_SIGMOID = True
VOLUME_RESTRICT = [8, 10, 10]
SILHOUETTE_LOSS_FUNC = 'L1'  # L1 or MSE
IS_FIX_VOLUME = False

# Volumetric Primitives
CUBOID_NUM = 8
SPHERE_NUM = 8
CONE_NUM = 0

# Visualize
EXPERIMENT_PATH = 'experiment'
SHOW_DIST = 2
TENSORBOARD_PATH = '/home/hank/3d/Tensorboard'
EXPERIMENT_NAME = 'view_center_add_can_cd_loss'

# Dataset
DATASET_ROOT = '/eva_data/hank'
IS_VIEW_CENTER = False
IS_NORMALIZE = False
LITTLE_NUM = {'train': 100, 'test': 20}
IMG_SIZE = 128

# airplane, rifle, display, table, telephone, car, chair, bench, lamp, cabinet, loudspeaker, sofa, watercraft
TRAIN_CLASSES = ['airplane', 'car', 'chair']
TEST_CLASSES = ['airplane', 'car', 'chair']
# TEST_CLASSES = ['rifle', 'display', 'table', 'telephone',
# 'bench', 'lamp', 'cabinet', 'loudspeaker', 'sofa', 'watercraft']
