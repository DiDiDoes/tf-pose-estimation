import argparse
import logging
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


def ckpt_tensors(ckpt):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if "efficient" in key:
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))


if __name__ == '__main__':
    #ckpt = '/data/models/baseline-spot4f/model_latest-'
    #ckpt = './models/pretrained/efficientnet-b0/model.ckpt'
    ckpt = tf.train.latest_checkpoint('/data/models/test/')
    print(ckpt)
    #ckpt_tensors(ckpt)

    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w <= 0 or h <= 0:
        w = h = None
    print(w, h)
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')

    net, pretrain_path, last_layer = get_network(args.model, input_node, None, trainable=False)
    '''
    if args.quantize:
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)
    '''
    with tf.Session(config=config) as sess:
        var_list = {v.op.name: v for v in tf.global_variables()}
        loader = tf.train.Saver(var_list)
        loader.restore(sess, ckpt)

        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile('/data/models/test.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)
    '''
        tf.train.write_graph(sess.graph_def, './tmp', 'graph.pb', as_text=True)

        flops = tf.profiler.profile(None, cmd='graph', options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP = ', flops.total_float_ops / float(1e6))

        # graph = tf.get_default_graph()
        # for n in tf.get_default_graph().as_graph_def().node:
        #     if 'concat_stage' not in n.name:
        #         continue
        #     print(n.name)

        saver = tf.train.Saver(max_to_keep=100)
        saver.save(sess, './tmp/chk', global_step=1)

    saver = tf.train.import_meta_graph('./tmp/chk-1.meta')

    #config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver.restore(sess, './tmp/chk-1')

        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()

    '''


