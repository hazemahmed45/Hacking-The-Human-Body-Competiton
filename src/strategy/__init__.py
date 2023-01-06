from src.config import BaseConfigs
from pytorch_lightning.strategies.bagua import BaguaStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.ddp2 import DDP2Strategy
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.strategies.dp import DataParallelStrategy
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy
from pytorch_lightning.strategies.horovod import HorovodStrategy
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.sharded import DDPShardedStrategy
from pytorch_lightning.strategies.sharded_spawn import DDPSpawnShardedStrategy
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from ray_lightning import RayShardedStrategy,RayStrategy
# from ray_lightning.tune import 
from pytorch_lightning.strategies.strategy import Strategy
from src.enums import StrategyTypes
import torch


def get_strategy(config:BaseConfigs) -> Strategy:
    """
    strategy builder function, that get the correct strategy specificed in the config class
    :param config: object of the config class
    :type config: BaseConfigs
    :return: pytorch_lightning.strategies.strategy.Strategy
    """
    num_gpu=[torch.device(i) for i in range(torch.cuda.device_count())] if config.strategy_type!=StrategyTypes.SINGLE_DEVICE else 0
    if(config.strategy_type == StrategyTypes.BAGUA):
        return BaguaStrategy()
    elif(config.strategy_type == StrategyTypes.CPU):
        return SingleDeviceStrategy(device='cpu')
    elif(config.strategy_type == StrategyTypes.DATA_PARALLEL):
        return DataParallelStrategy(accelerator=config.accelerator,parallel_devices=num_gpu)
    elif(config.strategy_type == StrategyTypes.DEEP_SPEED):
        return DeepSpeedStrategy(accelerator=config.accelerator)
    elif(config.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL):
        return DDPStrategy(accelerator=config.accelerator,parallel_devices=num_gpu)
    elif(config.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_2):
        return DDP2Strategy(accelerator=config.accelerator,parallel_devices=num_gpu)
    elif(config.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_FULLY_SHARDED):
        return DDPFullyShardedStrategy(accelerator=config.accelerator)
    elif(config.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_SHARDED):
        return DDPShardedStrategy(accelerator=config.accelerator)
    elif(config.strategy_type == StrategyTypes.DISTRIBUTED_DATA_PARALLEL_SPAWN_SHARDED):
        return DDPSpawnShardedStrategy(accelerator=config.accelerator)
    elif(config.strategy_type == StrategyTypes.HOROVOD):
        return HorovodStrategy(accelerator=config.accelerator,parallel_devices=num_gpu)
    elif(config.strategy_type == StrategyTypes.PARALLEL):
        return ParallelStrategy(accelerator=config.accelerator,parallel_devices=num_gpu)
    elif(config.strategy_type == StrategyTypes.SINGLE_DEVICE):
        return SingleDeviceStrategy(accelerator=config.accelerator)
    elif(config.strategy_type == StrategyTypes.RAY):
        return RayStrategy(resources_per_worker={'CPU':4,'GPU':1})#use_gpu=config.accelerator=='gpu',num_workers=config.num_workers,num_gpu=num_gpu[0])
    elif(config.strategy_type == StrategyTypes.RAY_SPAWN):
        return RayShardedStrategy(accelerator=config.accelerator)
    else:
        raise NotImplementedError(f"this strategy {config.strategy_type} is not yet implemented") 