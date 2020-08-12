import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow


def ckpt_tensors(ckpt):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if "efficient" in key:
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))


if __name__ == '__main__':
    # ckpt = '/data/models/baseline-spot4f/model_latest-'
    ckpt = './models/pretrained/efficientnet-b0/model.ckpt'
    #ckpt = tf.train.latest_checkpoint('/data/models/baseline-spot4m/')
    ckpt_tensors(ckpt)

    saver = tf.train.import_meta_graph(ckpt + '.meta')

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver.restore(sess, ckpt)

        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        summary_write = tf.summary.FileWriter("./" , graph)
        summary_write.close()
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile('/data/models/baseline-spot4m.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)


