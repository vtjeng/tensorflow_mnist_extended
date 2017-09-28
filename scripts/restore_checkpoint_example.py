import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

"""
Example code to show how to extract saved Variable values from a checkpoint.
"""
save_UUID = "2017-09-26_170603"
checkpoint_number = 703000
save_name = os.path.join("checkpoints", save_UUID, "cp-{}".format(checkpoint_number))
reader = tf.pywrap_tensorflow.NewCheckpointReader(save_name)
var_to_shape_map = reader.get_variable_to_shape_map()
print(var_to_shape_map["conv2/weight"])