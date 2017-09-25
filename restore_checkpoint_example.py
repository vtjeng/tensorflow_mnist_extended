import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

save_UUID = "2017-09-25_165023"
checkpoint_number = 0
save_name = os.path.join("checkpoints", save_UUID, "cp-{}".format(checkpoint_number))

def main(_):

    saver = tf.train.import_meta_graph("{}.meta".format(save_name))

    with tf.Session() as sess:
        saver.restore(sess, save_name)
        print_tensors_in_checkpoint_file(file_name=save_name, tensor_name='', all_tensors=True)
        print(tf.global_variables())


if __name__ == '__main__':
    tf.app.run()