from src.config import BaseConfigs
from src.callbacks import get_callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from src.trainer import WithDataloaderSegmentationTrainer
from pytorch_lightning.trainer import Trainer,seed_everything
import torch
from src.enums import ConfigTypes
from src.strategy import get_strategy
import os
import torch
import numpy as np
import random

def seeding(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic = True
    return 
if __name__ == '__main__':
    config=BaseConfigs()
    if(config.deterministic):
        seed_everything(config.random_seed,workers=True)
        seeding(config.random_seed)
    experiment_name=config.get_exp_name()
    config.create_exp_dir(experiment_name)
    logger=TensorBoardLogger(
        save_dir=config.get_exp_dir(),
        name=experiment_name,
        prefix=config.config_type.value if config.config_type==ConfigTypes.TRAIN else os.path.join(config.config_type.value,config.get_dataset_name()),
        log_graph=True
    )
    callbacks=get_callbacks(config)
    
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.epochs,
        strategy=get_strategy(config=config),
        replace_sampler_ddp=True,
        sync_batchnorm=True,
        log_every_n_steps=1,
        )
    config.effective_batch_size=config.batch_size*trainer.num_devices*trainer.num_nodes
    trainer_module=WithDataloaderSegmentationTrainer(config)
    trainer.fit(trainer_module)
