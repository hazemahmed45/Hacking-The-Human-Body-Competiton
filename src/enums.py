from enum import Enum
import os

class ModelTypes(Enum):
    """
    Types of the model architectures
    """
    
    UNET = 'UNET_FULL'
    CONDITIONAL_GUIDED_UNET_OUT_MASK_OUT_CLASS = 'CONDITIONAL_GUIDED_UNET_OUT_MASK_OUT_CLASS'
    UNET_LIGHT = 'UNET_LIGHT'
    CONDITIONAL_GUIDED_UNET_OUT_MASK_IN_CLASS = 'CONDITIONAL_GUIDED_UNET_OUT_MASK_IN_CLASS'

class ModelBackboneTypes(Enum):
    """
    Types of the model backbone architecture
    """
    NONE = 'none'
    UNET = 'unet_encoder'
    AUTOENCODER = 'autoencoder_encoder'
    HOURGLASS = 'hourglass_encoder'
    WNET = 'wnet_encoder'

class DatasetTypes(Enum):
    """
    Types of the datasets
    """
    NONE = 'none'
    SEGMENTATION = 'seg'
    SEGMENTATION_BINARY = 'seg_binary'
    SEGMENTATION_BINARY_WITH_CLASS = 'seg_binary_with_class'
    SEGMENTATION_BINARY_IMAGE_ORGAN = 'seg_binary_image_organ'



class AugmentationTypes(Enum):
    """
    Types of the augmentation pipelines
    """
    ALBUMENTATION_HEAVY = 'aug_heavy'
    ALBUMENTATION_LIGHT = 'aug_light'
    ALBUMENTATION_NONE = 'aug_none'



class EnviromentVariables(Enum):
    """
    Enviromental variables for all repo
    """
    TRAINING_UNIVERSE_RAW_DATA_LOCATION = os.path.join(os.path.abspath(''),'data','raw_datasets')
    TRAINING_UNIVERSE_TILED_DATA_LOCATION = os.path.join(os.path.abspath(''),'data','tiled_datasets')
    TRAINING_UNIVERSE_REMOTE_SUFFIX = r'Training-Universe/'
    BASE_EXPERIMENTS_DIR= os.path.abspath('exp')

class CriterionTypes(Enum):
    """
    Types of the criterions
    """
    CROSS_ENTROPY = 'ce'
    CROSS_ENTROPY_WITH_CLASS = 'ce_with_class'
    FOCAL = 'focal'

class OptimizerTypes(Enum):
    """
    Types of the optimizers
    """
    ADAM = 'adam'
    SGD = 'sgd'


class MetricTypes(Enum):
    """
    Types of the metrics
    """
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1SCORE = 'f1score'
    LOSS = 'loss'
    IOU = 'iou'
    DICE = 'dice'
    STATS = 'stats'
    MASK_ORGAN_STATS = 'mask_organ_stats'
    MASK_ORGAN_IOU = 'mask_organ_iou'
    MASK_ORGAN_DICE = 'mask_organ_dice'


class SchedularTypes(Enum):
    """
    Types of the schedulars
    """
    STEP = 'step'
    Linear = 'linear'
    LAMBDA = 'lambda'
    CONSINE_ANNEALING = 'cosine_annealing'
    EXPONENTIAL = 'exponential'
    COSINE_ANNEALING_WARM_START = 'cosine_annealing_warm_start'
    MULTISTEP = 'multistep'
    CONSTANT = 'constant'
    REDUCE_ON_PLATEAU = 'reduce_on_plateau'
    SEQUENTIAL = 'sequential'
    CYCLIC = 'cyclic'
    CHAINED = 'chained'


class ConfigTypes(Enum):
    """
    Types of the Configs
    """
    BASE = 'None'
    TRAIN = 'TRAINING'
    EVALUTAION = 'EVALUTAION'
    INFERENCE = 'INFERENCE'


class LoggerTypes(Enum):
    """
    Types of the loggers
    """
    TENSORBOARD = 'tensorboard'
    WANDB = 'wandb'



class CallbackTypes(Enum):
    """
    Types of the callbacks
    """
    MODEL_CHECKPOINT = 'model_checkpoint'
    EARLY_STOPPING = 'early_stopping'
    GRADIENT_ACCUMULATION_SCHEDULAR = 'gradient_accumulation_schedular'
    TQDM_PROGRESS_BAR = 'tqdm_progress_bar'
    DEVICE_STATS_MONITOR = 'device_stats_monitor'
    GPUS_STATS_MONITOR = 'gpus_stats_monitor'
    TIMER = 'timer'
    LEARNING_RATE_MONITOR = 'learning_rate_monitor'
    MODEL_SUMMARY = 'model_summary'
    RICH_MODEL_SUMMARY = 'rich_model_summary'
    RAY_TUNER_REPORT = 'ray_tuner_report'
    RAY_TUNER_CHECKPOINT_REPORT = 'ray_tuner_checkpoint_report'
    QUANTIZATION_CALIBRATOR = 'quantization_calibrator'



class SamplerTypes(Enum):
    """
    Types of the samplers
    """
    NONE_SAMPLER = 'none'
    BATCH_SAMPLER = 'batch'
    DISTRIBUTED_SAMPLER = 'distributed'
    RANDOM_SAMPLER = 'random'
    SEQUNETIAL_SAMPLER ='sequential'
    SUBSET_RANDOM_SAMPLER = 'subset_random'
    WEIGHTED_RANDOM_SAMPLER = 'weighted_random'


class StrategyTypes(Enum):
    """
    Types of the strategies
    """
    DISTRIBUTED_DATA_PARALLEL = 'ddp'
    DISTRIBUTED_DATA_PARALLEL_2 = 'ddp2'
    DATA_PARALLEL = 'dp'
    BAGUA = 'bagua'
    DISTRIBUTED_DATA_PARALLEL_FULLY_SHARDED = 'ddp_fully_sharded'
    DISTRIBUTED_DATA_PARALLEL_SPAWN_SHARDED = 'ddp_sharded_spawn'
    DISTRIBUTED_DATA_PARALLEL_SHARDED = 'ddp_sharded'
    DEEP_SPEED = 'deepspeed'
    HOROVOD = 'horovod'
    PARALLEL = 'parallel'
    SINGLE_DEVICE = 'single_device'
    CPU = 'cpu'
    RAY = 'ray'
    RAY_SPAWN = 'ray_spawn'


class MonitorModeTypes(Enum):
    """
    Types of the monitor mode
    """
    MIN = 'min'
    MAX = 'max'

class DictKeys(Enum):
    """
    constants in all the repo
    """
    MODEL_STATE_DICT = 'model_state_dict'
    BEST_SCORE = 'best_score'
    Y_PRED = 'y_pred'
    Y_TRUE = 'y_true'
    INPUT = 'input'
    SEGMENTATION = 'segmentation'
    TRAIN = 'train'
    VALID = 'val'
    TEST = 'test'
    BATCH_LOSS = 'batch_loss'
    IMAGE = 'image'
    MASK = 'mask'
    FILENAMES = 'filenames'
    IMG='img'
    LBL='lbl'
    DATASET_DIR='dataset_dir'
    CKPT ='ckpt'
    DATA = 'data'
    METRICS = 'metrics'
    CONFIG = 'config'



class SegmentationClassNames(Enum):
    """
    Class names
    """
    BACKGROUND = 'background'
    FUNTIONAL_TISSUE_UNIT = 'ftu'

