import os
from os.path import dirname, abspath

import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

from network_mobilenet import MobilenetNetwork
from network_mobilenet_thin import MobilenetNetworkThin

from network_cmu import CmuNetwork
from network_mobilenet_v2 import Mobilenetv2Network
from network_efficientnet import EfficientnetNetwork
from network_efficientdet import EfficientdetNetwork
from network_efficientdet2 import EfficientdetNetwork2


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_fast':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_accurate':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=1.00, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type in ['mobilenet_v2_w1.4_r1.0', 'mobilenet_v2_large', 'mobilenet_v2_large_quantize']:       # m_v2_large
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, conv_width2=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.4_r0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r1.0':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w0.75_r0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.75, conv_width2=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w0.5_r0.5' or type == 'mobilenet_v2_small':                                # m_v2_fast
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_v2_1.4':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_1.0':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type in ['cmu', 'openpose']:
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type in ['cmu_quantize', 'openpose_quantize']:
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'train/cmu/bs8_lr0.0001_q_e80/model_latest-18000'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'efficientnet-b0':
        net = EfficientnetNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'pretrained/efficientnet-b0/model.ckpt'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'efficientdet-d0':
        net = EfficientdetNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'pretrained/efficientdet-d0/model'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'efficientdet2':
        net = EfficientdetNetwork2({'image': placeholder_input}, trainable=trainable)
        pretrain_path_full = 'pretrained/efficientdet-d0/model'
        last_layer = 'Mconv7_stage6_L{aux}'

    else:
        raise Exception('Invalid Model Name.')

    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
    if sess_for_load is not None:
        if type in ['cmu', 'vgg', 'openpose']:
            if not os.path.isfile(pretrain_path_full):
                raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        else:
            try:
                s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            except:
                s = ''
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-246038' % s,
                'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
                'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000',
                'mobilenet_v2_w1.4_r0.5': 'trained/mobilenet_v2_w1.4_r0.5/model_latest-380401',
                'mobilenet_v2_large': 'trained/mobilenet_v2_w1.4_r1.0/model-570000',
                'mobilenet_v2_small': 'trained/mobilenet_v2_w0.5_r0.5/model_latest-380401',
            }
            ckpt_path = os.path.join(_get_base_path(), ckpts[type])
            loader = tf.train.Saver()
            try:
                loader.restore(sess_for_load, ckpt_path)
            except Exception as e:
                raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))

    return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': 'graph/cmu/graph_opt.pb',
        'openpose_quantize': 'graph/cmu/graph_opt_q.pb',
        'mobilenet_thin': 'graph/mobilenet_thin/graph_opt.pb',
        'mobilenet_0.5': 'pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt',
        'mobilenet_v2_large': 'graph/mobilenet_v2_large/graph_opt.pb',
        'mobilenet_v2_large_r0.5': 'graph/mobilenet_v2_large/graph_r0.5_opt.pb',
        'mobilenet_v2_large_quantize': 'graph/mobilenet_v2_large/graph_opt_q.pb',
        'mobilenet_v2_small': 'graph/mobilenet_v2_small/graph_opt.pb',
        'efficientnet-b0': '/data/models/backbone.pb',
        'efficientdet-d0': '/data/models/baseline-spot4m.pb',
        'efficientdet2': '/data/models/baseline-densefuse.pb'
    }

    base_data_dir = dirname(dirname(abspath(__file__)))
    if os.path.exists(os.path.join(base_data_dir, 'models')):
        base_data_dir = os.path.join(base_data_dir, 'models')
    else:
        base_data_dir = os.path.join(base_data_dir, 'tf_pose_data')

    graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
    return graph_path
    if os.path.isfile(graph_path):
        return graph_path

    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))
    return flops, params

def print_params(constant_values):
    total = 0
    prompt = []
    keyword = ['efficientnet','Openpose']
    for k,v in constant_values.items():
        # filtering some by checking ndim and name
        if v.ndim<1: continue
        if v.ndim==1:
            token = k.split(r'/')[-1]
            flag = False
            for word in keyword:
                if token.find(word)==-1:
                    flag = True
                    break
            if not flag:
                continue

        shape = v.shape
        cnt = 1
        for dim in shape:
            cnt *= dim
        prompt.append('{} with shape {} has {}'.format(k, shape, cnt))
        print(prompt[-1])
        total += cnt
    prompt.append('totaling {}'.format(total))
    print(prompt[-1])
    return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='efficientnet-b0', help='model name')
    parser.add_argument('--graph', default='cmu', help='graph name')
    args = parser.parse_args()
    
    print('Creating model: ', args.model)
    input_node = tf.placeholder(tf.float32, shape=(1, 384, 384, 3), name='image')
    net, pretrain_path, last_layer = get_network(args.model, input_node)

    pb = args.graph + '.pb'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        var_list = net.restorable_variables()
        
        try:
            print('Restore all weights.')
            loader = tf.train.Saver(net.restorable_variables(only_backbone=False))
            loader.restore(sess, pretrain_path)
            print('Restore all weights...Done')
        except:
            print('Restore only weights in backbone layers.')
            loader = tf.train.Saver(net.restorable_variables())
            loader.restore(sess, pretrain_path)
            print('Restore pretrained weights...Done')
        
        graph = tf.get_default_graph()

        print('stats before freezing')
        flops1, params1 = stats_graph(graph)

        graph_def = graph.as_graph_def()

        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile(pb, 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)

        g2 = load_pb(pb)
        g2.as_default()
        print('stats after freezing')
        flops2, params2 = stats_graph(g2)
        
        print('graph saved to ', pb)
