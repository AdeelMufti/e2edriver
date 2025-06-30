# config.py
from enum import Enum
import os

import torch

from utils import get_console_logger

logger = get_console_logger()
USER = os.environ.get("USER") or os.environ.get("USERNAME") or ""


### Data ###
class DataloaderType(Enum):
    """Enum for different data loader types."""

    # Each type has a tuple of (index, percentage of data)
    # The percentage is used to split the data into different sets
    TRAIN = 0, 1.0
    VALIDATION_HOLDOUT = 1, 0.0
    TEST_HOLDOUT = 2, 0.0
    VALIDATION = 3, 1.0
    TEST = 4, 1.0


class DataColumn(Enum):
    NAME = 0, "name"
    PAST_STATES_POS = 1, "past_states_pos"
    PAST_STATES_VEL = 2, "past_states_vel"
    PAST_STATES_ACCEL = 3, "past_states_accel"
    FUTURE_STATES_POS = 4, "future_states_pos"
    INTENT = 5, "intent"
    IMAGE_1_FRONT = 6, "IMAGE_1_FRONT"  # [269, 243, 3]
    IMAGE_2_FRONT_LEFT = 7, "IMAGE_2_FRONT_LEFT"  # [269, 243, 3]
    IMAGE_3_FRONT_RIGHT = 8, "IMAGE_3_FRONT_RIGHT"  # [269, 243, 3]
    IMAGE_4_SIDE_LEFT = 9, "IMAGE_4_SIDE_LEFT"  # [269, 243, 3]
    IMAGE_5_SIDE_RIGHT = 10, "IMAGE_5_SIDE_RIGHT"  # [269, 243, 3]
    IMAGE_6_REAR_LEFT = 11, "IMAGE_6_REAR_LEFT"  # [146, 243, 3 ]
    IMAGE_7_REAR = 12, "IMAGE_7_REAR"  # [137, 243, 3]
    IMAGE_8_REAR_RIGHT = 13, "IMAGE_8_REAR_RIGHT"  # [146, 243, 3]


class ProcessedDataColumn(Enum):
    NAME = 0
    INTENT = 1
    PAST_STATES = 2
    FUTURE_STATES = 3
    IMAGES = 4
    STATE_EMBEDDING = 5
    IMAGE_EMBEDDING = 6
    COMBINED_EMBEDDING = 7


class PredictedDataColumn(Enum):
    FUTURE_STATES = 0
    FUTURE_STATES_GMM = 1  # Tuple of (coef, mu, ln_var) tensors
    FUTURE_STATES_GMM_SAMPLED = 2
    FUTURE_STATES_MNLL = 3  # Tuple of (mode_probability, predicted_trajectory) tensors
    FUTURE_STATES_MNLL_SAMPLED = 4


TRAIN_DATASET_DIR = (
    f"/home/{USER}/data/waymo_open_dataset_end_to_end_camera_v_1_0_0_parquet_v0_1"
)
VALIDATION_DATASET_DIR = (
    f"/home/{USER}/data/waymo_open_dataset_end_to_end_camera_v_1_0_0_parquet_v0_1_val"
)
TEST_DATASET_DIR = (
    f"/home/{USER}/data/waymo_open_dataset_end_to_end_camera_v_1_0_0_parquet_v0_1_test"
)
COLUMNS = [column.value[1] for column in DataColumn]
logger.info(f"Data columns: {COLUMNS}")

NUM_DATA_LOADERS = 10  # Number of subprocesses for data loading


# Data Format Parameters #
HERTZ = 4  # Frequency of data collection (4Hz)
NUM_PAST_SECONDS = 4
NUM_FUTURE_SECONDS = 5
PAST_STATE_DIM = 2 * (1 + 1 + 1)  # xy for position, acceleration, and velocity
FUTURE_STATE_DIM = 2  # xy for future position
NUM_PAST_STATES = (
    PAST_STATE_DIM * NUM_PAST_SECONDS * HERTZ
)  # Number of past states to consider (4s x 4Hz)
NUM_FUTURE_STATES = (
    NUM_FUTURE_SECONDS * HERTZ * FUTURE_STATE_DIM
)  # Number of future states to predict (5s x 4Hz)


### Logging and Checkpoints ###
EXPERIMENT_NAME = "default_experiment"  # Name of the experiment for logging
LOG_DIR = f"/home/{USER}/data/logs"
CHECKPOINT_DIR = f"/home/{USER}/data/checkpoints"
SAVE_CHECKPOINT_FREQ = 1  # Save checkpoint every N epochs
LOG_STEP_FREQ = 10  # Log training loss every N steps


### Model parameters ###
# Standard WOD setup (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
NUM_CAMERAS = 8


# Training parameters
class TrainerStepType(Enum):
    TRAIN_STEP = 0
    VALIDATION_STEP = 1
    TRAIN_EPOCH = 2
    VALIDATION_EPOCH = 3


DTYPE = torch.float32
USE_AMP = True  # Automatic Mixed Precision

SHUFFLE_TRAIN = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # Adjust based on your GPU memory

LEARNING_RATE = 1e-5
STEP_LR_STEP_SIZE = 1  # Epochs
STEP_LR_GAMMA = 0.9
ONE_CYCLE_MAX_LR = LEARNING_RATE * 15.0
ONE_CYCLE_LR_DIV_FACTOR = 10
ONE_CYCLE_FINAL_DIV_FACTOR = 1000
GRADIENT_CLIP_MAX_NORM = 0.0  # None or 0.0 to disable

NUM_EPOCHS = 5

EARLY_STATE_FUSION = True
IMAGE_RESNET_FINAL_MAP_DIM = 16
VIT_DEPTH = 2
VIT_NUM_HEADS = 2
VIT_MLP_RATIO = 2.0
USE_IMAGE_FUSION_TRANSFORMER = True
IMAGE_FUSION_TRANSFORMER_HEADS = 2
IMAGE_FUSION_TRANSFORMER_FF_DIM = 1024
IMAGE_FUSION_TRANSFORMER_LAYERS = 2
IMAGE_EMBEDDING_DIM = 256

STATE_EMBEDDING_DIM = 128
STATE_LAYERS = 3
STATE_LAYERS_DROPOUT = 0.01

INTENT_EMBEDDING_DIM = 16  # Dimension of intent feature vector
INTENT_LAYERS = 3
INTENT_LAYERS_DROPOUT = 0.01
INTENT_CLASSES = 4  # waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_data.proto#EgoIntent

TRAJECTORY_DECODER_EMBEDDING_DIM = 256
TRAJECTORY_DECODER_LAYERS = 3  # Number of layers in trajectory decoder
TRAJECTORY_DECODER_LAYERS_DROPOUT = 0.01

GMM_TRAJECTORY_DECODER_ENABLED = True  # Use GMM for trajectory prediction
GMM_COMPONENTS = INTENT_CLASSES  # Number of GMM components
GMM_TRAJECTORY_DECODER_EMBEDDING_DIM = 256
GMM_TRAJECTORY_DECODER_LAYERS = 3
GMM_TRAJECTORY_DECODER_LAYERS_DROPOUT = 0.01

MNLL_TRAJECTORY_DECODER_ENABLED = True  # Use (Multimodal) Minimum Negative Log Likelihood (MNLL) for trajectory prediction
MNLL_MODES = INTENT_CLASSES  # Number of modes for MNLL
MNLL_TRAJECTORY_DECODER_EMBEDDING_DIM = 256
MNLL_TRAJECTORY_DECODER_LAYERS = 3
MNLL_TRAJECTORY_DECODER_LAYERS_DROPOUT = 0.01


### Losses ###
class Loss(Enum):
    MSETrajectoryLoss = 0
    L1TrajectoryLoss = 1
    GMMTrajectoryLoss = 2
    MNLLTrajectoryLoss = 3


class LossConfig(Enum):
    NAME = 0
    FUNCTION = 1
    WEIGHT = 2
    PARAMS = 3


LOSSES = [
    {
        LossConfig.NAME: Loss.MSETrajectoryLoss,
        LossConfig.WEIGHT: 1.0,
        LossConfig.PARAMS: {},  # No specific params needed for MSETrajectoryLoss beyond default name
    },
]
if GMM_TRAJECTORY_DECODER_ENABLED:
    LOSSES.append(
        {
            LossConfig.NAME: Loss.GMMTrajectoryLoss,
            LossConfig.WEIGHT: 1.0,
            LossConfig.PARAMS: {
                "num_components": GMM_COMPONENTS,  # Number of GMM components
            },
        },
    )
if MNLL_TRAJECTORY_DECODER_ENABLED:
    LOSSES.append(
        {
            LossConfig.NAME: Loss.MNLLTrajectoryLoss,
            LossConfig.WEIGHT: 1.0,
            LossConfig.PARAMS: {
                "num_modes": MNLL_MODES,  # Number of modes for MNLL
            },
        },
    )

assert len(LOSSES) > 0, "No loss functions specified. Please check the configuration."


### Misc ###
CHECKPOINT_STATE_VARIABLES = set(
    [
        "epoch",
        "global_step",
        "best_metric",
        "state_dict",
        "optimizer",
        "scaler",
        "scheduler",
    ]
)
