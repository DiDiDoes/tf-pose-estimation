import tensorflow as tf
from network_efficientnet import EfficientnetNetwork
from tensorflow.python.framework import graph_util

if __name__ == '__main__':
    ckpt = tf.train.latest_checkpoint('../models/train/eff0.5/')

    input = tf.placeholder(tf.float32, shape=(None, 384, 384, 3), name='image')
    network = EfficientnetNetwork({'image': input})

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)

        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile('../models/graph/efficientnet-b0/graph_opt.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)


