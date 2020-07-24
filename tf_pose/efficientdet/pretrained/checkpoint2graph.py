import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

if __name__ == '__main__':
    ckpt = 'efficientdet-d0/model'
    
    if False:
        reader=pywrap_tensorflow.NewCheckpointReader(ckpt)
        var_to_shape_map=reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print('tensor_name: ', key)

    saver = tf.train.import_meta_graph(ckpt + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt)

        graph_def = tf.get_default_graph().as_graph_def()

        node_list=[n.name for n in graph_def.node]
        for node in node_list:
            if 'save' not in node and 'ExponentialMovingAverage' not in node and 'cond' not in node and 'Initialize' not in node:
                print(node)

        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['box_net/Sigmoid_14', 'class_net/Sigmoid_14']
                )

        with tf.gfile.GFile('efficientdet-d0/graph_opt.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)

        flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOPs: {}'.format(flops.total_float_ops))
