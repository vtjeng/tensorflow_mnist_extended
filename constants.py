import os

BATCH_SIZE = 50
NUM_EPOCHS = 1


"""
Constants for logging and storing data. You are unlikely to want to modify these.
"""
PWD = os.path.dirname(__file__) + '/'
CHECKPOINT_DIR = PWD + 'checkpoints/'
LOGS_DIR = PWD + 'logs/'
TB_LOGS_DIR = PWD + 'tb_logs/'
