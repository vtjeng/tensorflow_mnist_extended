import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import scipy.io as sio
import numpy as np

save_UUID = "2017-09-28_183158"


save_name = os.path.join("checkpoints", save_UUID, "cp_cleverhans")
reader = tf.pywrap_tensorflow.NewCheckpointReader(save_name)
var_to_shape_map = reader.get_variable_to_shape_map()
print(var_to_shape_map)
sio.savemat(os.path.join("checkpoints", '{save_UUID}-ch-params.mat'.format(save_UUID = save_UUID)),
            {k: reader.get_tensor(k) for k in var_to_shape_map.keys()})