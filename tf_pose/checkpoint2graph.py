import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow


def ckpt_tensors(ckpt):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


if __name__ == '__main__':
    # ckpt = './models/pretrained/efficientdet-d0/model'
    ckpt = tf.train.latest_checkpoint('/data/models/baseline-spot4m/')
    # ckpt_tensors(ckpt)

    saver = tf.train.import_meta_graph(ckpt + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt)

        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile('/data/models/baseline-spot4m.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)


