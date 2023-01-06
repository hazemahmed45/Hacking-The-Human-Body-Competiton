import os
import json
from typing import Dict
from src.enums import *

# Config class that holds all logger configurations
class LoggerConfigs():
    """
    Configuration class specified for loggers
    """
    def __init__(self) -> None:
        """
        configuration for loggers
        - logger_type : which type of logger to using during training
        - workspace_name : name of the workspace the logger use in its initialization
        - project_name : name of the project to log to
        - run_desc : (optional) description of the project 
        - resume_run : name of the run to continue from where you stopped (depends on entrypoint used)
        """
        self.logger_type=LoggerTypes.TENSORBOARD
        self.workspace_name='hazem'
        self.project_name="Hacking-Human-Body"
        self.run_desc=''
        self.resume_run=''

# Strategy class that holds all training strategy configurations
class StrategyConfigs():
    """
    Configuration class specified for strategies
    """
    def __init__(self) -> None:
        """
        configuration for training strategy
        - strategy_type : which type of strategy to using during training
        - accelerator : type of accelerator, can be cuda, cpu , hpu, ipu, or tpu
        - device : device to run on whether cuda or cpu
        """
        self.strategy_type=StrategyTypes.DATA_PARALLEL
        if(self.strategy_type != StrategyTypes.CPU):
            self.accelerator='gpu'
            self.device='cuda'
        else:
            self.device='cpu'
            self.accelerator='cpu'
    
    def set_sampler_type(self,sampler_type:SamplerTypes=None)->SamplerTypes:
        """
        set the correct sampler according to the training strategy, return Distributed sampler type if 
        strategy is ddp and random sampler if is not set
        :param sampler_type: default value is None, the sampler to be set to if possibile
        :type sampler_type: src.configs.enums.SamplerTypes
        :return: correct sampler type
        :rtype: src.configs.enums.SamplerTypes
        """
        if(
            self.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL or
            self.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_2 or
            self.strategy_type == StrategyTypes.DEEP_SPEED or
            self.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_FULLY_SHARDED or
            self.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_SPAWN_SHARDED or
            self.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_SHARDED or
            self.strategy_type == StrategyTypes.HOROVOD or
            self.strategy_type == StrategyTypes.BAGUA 
        ):
            return SamplerTypes.DISTRIBUTED_SAMPLER
        elif(sampler_type is not None):
            return sampler_type
        return SamplerTypes.RANDOM_SAMPLER

# Model class that holds all model architecture configurations
class ModelConfigs():
    """
    Configuration class specified for models
    """
    def __init__(self) -> None:
        """
        configuration for Model architecture
        - batch_norm : add batch normalization layers in model architecture
        - out_channels : number of classification channels
        - in_channels : number of input image channels
        - dataset_type : set the type of dataset, is set from src.configs.enums.DatasetTypes
        - model_type : set the type of model architecture, is set from src.configs.enums.ModelTypes
        - backbone_type : set the type of model backbone, is set from src.configs.enums.ModelBackboneTypes
        - backbone_path : backbone ckeckpoint path
        - criterion_type : set the type of criterion, is set from src.configs.enums.CriterionTypes
        - metrics : list of MetricTypes to evaluate model performance
        - checkpoint_path : model checkpoint path
        - return_last_layer : (optional) for the WNET, whether to return the middle unet last layer or no
        - n_hidden_dim : (optional) for the WNET, the number of hidden layers that connect first and second UNet
        - focal_gamma : (optional) focal loss hyperparameter
        - focal_alpha : (optional) focal loss hyperparameter
        """
        self.batch_norm=True
        self.out_channels=2
        self.in_channels=3
        self.dataset_type=DatasetTypes.SEGMENTATION_BINARY_IMAGE_ORGAN
        self.model_type=ModelTypes.CONDITIONAL_GUIDED_UNET_OUT_MASK_OUT_CLASS
        self.backbone_type=ModelBackboneTypes.NONE
        self.backbone_path=None
        self.criterion_type=CriterionTypes.CROSS_ENTROPY
        self.metrics=[MetricTypes.LOSS,MetricTypes.MASK_ORGAN_STATS,MetricTypes.MASK_ORGAN_IOU,MetricTypes.MASK_ORGAN_DICE]

        self.checkpoint_path=r''


        if(CriterionTypes.FOCAL):
            self.focal_gamma=0
            self.focal_alpha=None

# callbacks class that holds all callbacks configurations
class CallbacksConfigs():
    """
    Configuration class specified for callbacks
    """
    def __init__(self) -> None:
        """
        configuration for Callbacks 
        - callbacks : list of the callbacks used during training
        - ckpt_monitor : metric callback to save model checkpoint on
        - save_last : save the model on the last epoch
        - ckpt_verbose : whether to show the checkpointing logs or no in terminal
        - save_top_k : the number of best models to save
        - ckpt_monitor_mode : mode to save on, whether min or max 
        - save_weight_only : save the weight of the model only without any metadata
        - gradient_accumlation_schedular_factor : frequency of the gradient accumlation 
        - early_stop_monitor : metric callback to apply early stopping
        - early_stop_verbose : whether to show the early stopping logs or no in terminal
        - early_stop_monitor_mode : mode to save on, whether min or max 
        - early_stop_min_delta : minimum change in the monitored quantity to qualify as an improvement
        - early_stop_patience : number of checks with no improvement after which training will be stopped
        """
        self.callbacks=[CallbackTypes.MODEL_CHECKPOINT,CallbackTypes.TQDM_PROGRESS_BAR]
        
        if(CallbackTypes.MODEL_CHECKPOINT in self.callbacks):
            self.ckpt_monitor=f'{DictKeys.VALID.value}/RunningLoss'
            self.save_last=False
            self.ckpt_verbose=True
            self.save_top_k=1
            self.ckpt_monitor_mode=MonitorModeTypes.MIN
            self.save_weight_only=False
        if(CallbackTypes.GRADIENT_ACCUMULATION_SCHEDULAR in self.callbacks):
            self.gradient_accumlation_schedular_factor={4:0.5}
        if(CallbackTypes.EARLY_STOPPING in self.callbacks):
            # self.early_stop_schedular_factor={4:0.5}
            self.early_stop_monitor=f'{DictKeys.VALID.value}/loss'
            self.early_stop_verbose=True
            self.early_stop_monitor_mode=MonitorModeTypes.MIN
            self.early_stop_min_delta=1e-2
            self.early_stop_patience=4


class BaseConfigs(LoggerConfigs,StrategyConfigs,ModelConfigs,CallbacksConfigs):
    """
    Configuration class for general trails configurations
    """
    def __init__(self) -> None:
        """
        General configurations
        - epoch : number of iterations per training
        - deterministic : whether the run is capabile of being reproduced
        - random_seed : the seed of any random initializations
        - dataset_path : local path of the dataset used for training (str or list)
        - pin_memory : the data loader will copy Tensors into device/CUDA pinned memory before returning them
        - num_workers : how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
        - aug_bounds : on the fly augmentation is applied with a probability, this is the probability range (low,high)
        - train_img_width : image width size during training
        - train_img_height : image height size during training
        - test_img_width : test width size during testing
        - test_img_height : test height size during testing
        - sampler_type : the sampling technique used during collecting data during training , is set from src.configs.enums.SamplerTypes
        - batch_size : the number of images per step
        - effective_batch_size : (set automaticly) during dp is [batch_size*num_gpu] , during ddp is [batch_size*num_gpu*num_node]
        - shuffle : whether to shuffle the samples every epoch or no
        - lr : learning rate
        - schedular_type : set the schedular that changes the learning rate during training, is set from src.configs.enums.SchedularTypes
        - augmentation_type : set the type of augmentation pipeline, is set from src.configs.enums.AugmentationType
        - optim_type : set the type of optimizer, is set from src.configs.enums.OptimizerTypes
        - file_manager_type : set the type of file manager, is set from src.configs.enums.FileManagerTypes
        """
        LoggerConfigs.__init__(self)
        ModelConfigs.__init__(self)
        CallbacksConfigs.__init__(self)
        StrategyConfigs.__init__(self)
        self.epochs=300
        self.deterministic=True
        self.random_seed=123
        self.dataset_path=r"C:\Users\HazemHaroun\hazem\Hacking_the_human_body_competition\data\splitted_dataset\competition_data_splitted"
        self.pin_memory=True
        self.num_workers=4
        self.aug_bounds=[0.3,0.5]
        self.train_img_width=224
        self.train_img_height=224
        self.test_img_width=224
        self.test_img_height=224
        self.sampler_type=self.set_sampler_type()
        self.batch_size=4
        self.effective_batch_size=self.batch_size #will be set automatically
        self.shuffle=True
        self.lr=1e-5
        self.class_labels_dict={
                SegmentationClassNames.BACKGROUND.value:0,
                SegmentationClassNames.FUNTIONAL_TISSUE_UNIT.value:1,
            }
        self.class_color_dict={SegmentationClassNames.BACKGROUND.value:0,SegmentationClassNames.FUNTIONAL_TISSUE_UNIT.value:255}
        self.organ_labels_dict={
            'kidney':0,
            'prostate':1,
            'largeintestine':2,
            'spleen':3,
            'lung':4
        }
        self.freeze = False
        self.n_classes=3
        self.schedular_type=SchedularTypes.CONSTANT
        self.augmentation_type=AugmentationTypes.ALBUMENTATION_LIGHT
        self.optim_type=OptimizerTypes.ADAM
        self.exp_name=self.create_experiment_name()
        self.set_config_type()
        self.create_exp_dir(self.exp_name)

    
    
    def get_config_dict(self):
        """
        this function return dictionary contains config attribute
        :return: config as dictionary
        :rtype: Dict
        """
        config_dict={}
        temp_config_dict=vars(self)
        for key,value in temp_config_dict.items():
            if(isinstance(value,Enum)):
                config_dict[key]=value.value
            elif(isinstance(value,list)):
                config_dict[key]=[]
                for item in value:
                    if(isinstance(item,Enum)):
                        config_dict[key].append(item.value)
                    else:
                        config_dict[key].append(item)    
            else:
                config_dict[key]=value
        return config_dict
    
    def load_config_dict(self,config_file):
        """
        this function loads config class attributes using dictionary
        :param config_dict: the word to run the check on to
        :type config_dict: Dict
        """
        config_dict={}
        with open(config_file,'r') as f:
            config_dict=json.load(f)
        self.set_attr_from_config_dict(config_dict)
    
    def set_attr_from_config_dict(self,config_dict:Dict):
        """
        this function make sure the key is in config attributes to set it
        :param config_dict: the word to run the check on to
        :type config_dict: Dict
        """
        for key,value in config_dict.items():
            if(key in dir(self)):
                self.__setattr__(key,value)
    
    
    def set_config_type(self):
        """
        this function sets the configuration type
        """
        self.config_type=ConfigTypes.TRAIN
        return 
    
    def get_dataset_name(self):
        """
        this function generate the dataset name from dataset_path
        :return: dataset name
        :rtype: str
        """
        dataset_name=''
        if(isinstance(self.dataset_path,list)):
            for data_dir in self.dataset_path:
                dataset_name+=os.path.basename(data_dir)+' '
            dataset_name.rstrip()
        else:
            dataset_name=os.path.basename(self.dataset_path)
        return dataset_name
    
    def create_exp_dir(self,exp_name):
        """
        this function make sure the key is in config attributes to set it
        :param exp_name: experiment name
        :type exp_name: str
        """
        if(not os.path.exists(EnviromentVariables.BASE_EXPERIMENTS_DIR.value)):
            os.mkdir(EnviromentVariables.BASE_EXPERIMENTS_DIR.value)
        if(self.config_type==ConfigTypes.EVALUTAION or self.config_type==ConfigTypes.INFERENCE):
            self.cur_exp_dir=os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,self.logger_type.value,self.config_type.value,self.get_dataset_name(),exp_name)
        else:
            self.cur_exp_dir=os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,self.logger_type.value,self.config_type.value,exp_name)
        if(not os.path.exists( self.cur_exp_dir)):
            os.makedirs(self.cur_exp_dir)
        return 
    
    def isCamelCaseWord(self,word:str)-> bool:
        """
        this function check if the word is camel case or no
        :param word: the word to run the check on to
        :type word: str
        :return: is camel case or no
        :rtype: bool
        """
        first=False
        second=False
        for ii,c in enumerate(word):
            if(ii==0):
                if(c==str.upper(c)):
                    first=True
            else:
                if(c==str.upper(c)):
                    second=True
        return first and second
    
    def getCamelCaseInitails(self,word:str)->str:
        """
        this function gets the camel case word initials
        :param word: the word to get its initials
        :type word: str
        :return: camel case initials
        :rtype: str
        """
        capitals=''
        for ii,c in enumerate(word):
            if(c==str.upper(c)):
                capitals+=c
        return c[:2]
    
    def read_exp_no(self):
        if(not os.path.exists(os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,'exp_no.txt'))):
            with open(os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,'exp_no.txt'),'w')as f:
                f.write('0')
        with open(os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,'exp_no.txt'),'r') as f:
            exp_no=int(f.readline().replace("\n",''))
        with open(os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,'exp_no.txt'),'w') as f:
            f.write(str(exp_no+1))
        return exp_no
    
    def create_experiment_name(self,experiment_name=None):
        """
        this function creates and return experiment name
        :param experiment_name: (optional) return the same experiment name if given
        :type experiment_name: str
        :return: experiment name created
        :rtype: str
        """
        if(experiment_name is not None):
            self.exp_name=experiment_name
            return self.exp_name
        elif(self.resume_run != ""):
            self.exp_name=self.resume_run
            return self.exp_name
        if(self.project_name is None):
            import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            experiment_name=str(current_time)
        else:
            first_name=self.project_name.split('-')[0].split('_')[0].split('.')[0]
            print(first_name)
            if(self.isCamelCaseWord(first_name)):
                experiment_name=self.getCamelCaseInitails(first_name)
            elif(len(first_name)<=5):
                experiment_name=str.upper(first_name)
            else:
                experiment_name=str.upper(first_name[:3])
        exp_no=self.read_exp_no()
        self.exp_name=str.format('{}-{}',str(experiment_name),str(exp_no))
        return self.exp_name
    
    def get_exp_name(self):
        """
        this function returns the experiment name
        :return: experiment name
        :rtype: str
        """
        return self.exp_name
    
    def get_exp_dir(self):
        """
        this function returns the experiment dir
        :return: experiment directory
        :rtype: str
        """
        return os.path.join(EnviromentVariables.BASE_EXPERIMENTS_DIR.value,self.project_name,self.exp_name,self.config_type.value)





