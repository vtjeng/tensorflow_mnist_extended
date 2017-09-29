import os

"""
Parameters for neural network
"""
PREPROCESS_POOL = 2 # original value 1
WINDOW_1 = 3 # original value 5
WINDOW_2 = 3 # original value 5
POOL_1 = 2 # original value 2
POOL_2 = 1 # original value 2
NUM_CHANNELS_CONV1 = 4 # original value 32
NUM_CHANNELS_CONV2 = 8 # original value 64
NUM_CHANNELS_FC1 = 64 # original value 1024

"""
Parameters for training
"""
BATCH_SIZE = 50
NUM_EPOCHS = 640 # original value 20

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
