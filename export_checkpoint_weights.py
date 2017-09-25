import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import scipy.io as sio
import numpy as np

save_UUID = "2017-09-25_165023"
checkpoint_number = 0
save_name = os.path.join("checkpoints", save_UUID, "cp-{}".format(checkpoint_number))

def main(_):

    saver = tf.train.import_meta_graph("{}.meta".format(save_name))

    with tf.Session() as sess:
        saver.restore(sess, save_name)
        # print_tensors_in_checkpoint_file(file_name=save_name, tensor_name='', all_tensors=True)
        print(tf.global_variables())

        reader = tf.pywrap_tensorflow.NewCheckpointReader(save_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        print(var_to_shape_map["conv2/weight"])
        print(reader.get_tensor("conv2/weight"))

        x = np.zeros((1, 2, 3, 4))
        sio.savemat('zero.mat', {'zero': x})
        # sio.savemat('test.mat', {k: reader.get_tensor(k) for k in var_to_shape_map.keys()})


if __name__ == '__main__':
    tf.app.run()