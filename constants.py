import os

BATCH_SIZE = 50
NUM_EPOCHS = 20

EVAL_FREQUENCY = 100
CHECKPOINT_FREQUENCY = 1000
CHECKPOINT_MAX_KEEP = 5
CHECKPOINT_HOURS = 2


"""
Constants for logging and storing data. You are unlikely to want to modify these.
"""
PWD = os.path.dirname(__file__) + '/'
CHECKPOINT_DIR = PWD + 'checkpoints/'
LOGS_DIR = PWD + 'logs/'
TB_LOGS_DIR = PWD + 'tb_logs/'
