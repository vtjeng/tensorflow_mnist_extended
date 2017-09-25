import os

"""
Parameters for neural network
"""
NUM_CHANNELS_CONV1 = 4 # original value 32
NUM_CHANNELS_CONV2 = 8 # original value 64
NUM_CHANNELS_FC1 = 64 # original value 1024

"""
Parameters for training
"""
BATCH_SIZE = 50
NUM_EPOCHS = 320 # original value 20

"""
Parameters for monitoring training process + checkpointing
"""
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
