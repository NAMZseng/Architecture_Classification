from datetime import datetime

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
CHECKPOINT_PATH = 'checkpoint'
LOG_DIR = 'tensorboard'
RESULT_DIR = 'result'
