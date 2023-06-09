"""
Базовый конфигурационный файл.
Предлагается использовать YACS config (https://github.com/rbgirshick/yacs) для
поддержки базового конфигурационного файла. Затем для каждого конкретного
эксперимента можно создать конфигурационный файл, который будет переопределять
необходимые параметры базового конфигурационного файла.


Пример использования:
    1. Создать конфигурационный файл для конкретного эксперимента
    (например, configs/experiment_1.yaml)
    2. В конфигурационном файле переопределить необходимые параметры
    базового конфигурационного файла
    3. В модуле, где необходимо использовать конфигурационные параметры,
    импортировать функцию combine_config
    4. Вызвать функцию combine_config, передав в качестве аргумента путь
    к конфигурационному файлу конкретного эксперимента
    5. Полученный объект yacs CfgNode можно использовать для доступа
    к конфигурационным параметрам
"""

import os.path as osp
from typing import Union
from datetime import datetime
from yacs.config import CfgNode as CN


_C = CN()

# Root directory of project
_C.ROOT = CN()
_C.ROOT.PATH = '/Users/your_username/mfdp-sentiment'  # Путь к корневой директории проекта

# Dataset emotions
_C.DATASET_EMO = CN()
_C.DATASET_EMO.PATH = 'data/raw/emotions/six_emotions.csv'
_C.DATASET_EMO.SPLIT_OUTPUT_PATH = 'data/interim/emotions'
_C.DATASET_EMO.TEST_SIZE = 0.2
_C.DATASET_EMO.VAL_SIZE = 0.2
_C.DATASET_EMO.BALANCE_CLASSES = False

# Learning
# _C.LEARNING = CN()
# _C.LEARNING.EXPERIMENT = 'efficientnet-b0'
# _C.LEARNING.ETA = 3e-2
# _C.LEARNING.MAX_EPOCHS = 10
# _C.LEARNING.DEVICE = 'cpu'

# Logging
# _C.LOGGING = CN()
# _C.LOGGING.LOGS_FOLDER = 'logs'
# _C.LOGGING.EXPERIMENT_FOLDER = f'{datetime.now().strftime("%y_%m_%d_%H_%M")}'
# _C.LOGGING.MODEL_NAME = f'{_C.LEARNING.EXPERIMENT}.pt'
# _C.LOGGING.LOGGING_INTERVAL = 'step'

# MLFLOW

# _C.MLFLOW = CN()
# _C.MLFLOW.TRACKING_URI = ''
# _C.MLFLOW.S3_ENDPOINT_URL = ''

# Checkpoint
# _C.CHECKPOINT = CN()
# _C.CHECKPOINT.CKPT_FOLDER = 'checkpoints'
# _C.CHECKPOINT.FILENAME = '{epoch}_{valid_acc:.2f}_{valid_loss:.2f}'
# _C.CHECKPOINT.SAVE_TOP_K = 2
# _C.CHECKPOINT.CKPT_MONITOR = 'valid_loss'
# _C.CHECKPOINT.CKPT_MODE = 'min'

# Early stopping
# _C.ES = CN()
# _C.ES.MONITOR = 'valid_loss'
# _C.ES.MIN_DELTA = 2e-4
# _C.ES.PATIENCE = 10
# _C.ES.VERBOSE = False,
# _C.ES.MODE = 'min'

# test
# _C.TEST = CN()
# _C.TEST.SAVE = True



def get_cfg_defaults():
    """Возвращает yacs CfgNode объект со значениями по умолчанию"""
    return _C.clone()


def combine_config(cfg_path: Union[str, None] = None):
    """
    Объединяет базовый конфигурационный файл с
    конфигурационным файлом конкретного эксперимента
    Args:
         cfg_path (str): file in .yaml or .yml format with
         config parameters or None to use Base config
    Returns:
        yacs CfgNode object
    """
    base_config = get_cfg_defaults()
    if cfg_path is not None:
        if osp.exists(cfg_path):
            base_config.merge_from_file(cfg_path)
        else:
            raise FileNotFoundError(f'File {cfg_path} does not exists')

    # Join paths
    base_config.DATASET_EMO.PATH = osp.join(
        base_config.ROOT.PATH, 
        base_config.DATASET_EMO.PATH
    )
    base_config.DATASET_EMO.SPLIT_OUTPUT_PATH = osp.join(
        base_config.ROOT.PATH,
        base_config.DATASET_EMO.SPLIT_OUTPUT_PATH
    )

    return base_config
