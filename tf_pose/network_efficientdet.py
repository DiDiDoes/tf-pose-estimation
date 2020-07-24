from __future__ import absolute_import

import tensorflow as tf

import network_base
from efficientdet.efficientdet_arch import efficientdet
from network_base import layer

import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

class EfficientdetNetwork(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=0.5):
        self.conv_width = conv_width
        self.refine_width = conv_width2
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    @layer
    def base(self, input, name):
        net, endpoints = efficientdet(input, model_name=name)
        for k, tensor in sorted(list(endpoints.items()), key=lambda x: x[0]):
            self.layers['%s/%s' % (name, k)] = tensor
            print(k, tensor.shape)
        return net

    def setup(self):
        depth2 = lambda x: int(x * self.refine_width)

        self.feed('image').base(name='efficientdet-d0')

        # for n, l in enumerate(self.layers):
        #     print(l)

        self.feed('efficientdet-d0/P4').upsample(factor='efficientdet-d0/P3', name='efficientdet-d0/P4_upsample')
        self.feed('efficientdet-d0/P5').upsample(factor='efficientdet-d0/P3', name='efficientdet-d0/P5_upsample')
        self.feed('efficientdet-d0/P6').upsample(factor='efficientdet-d0/P3', name='efficientdet-d0/P6_upsample')
        self.feed('efficientdet-d0/P7').upsample(factor='efficientdet-d0/P3', name='efficientdet-d0/P7_upsample')

        (self.feed(
            'efficientdet-d0/P3',
            'efficientdet-d0/P4_upsample',
            'efficientdet-d0/P5_upsample',
            'efficientdet-d0/P6_upsample',
            'efficientdet-d0/P7_upsample',
        ).concat(3, name='feat_concat'))

        feature_lv = 'feat_concat'
        with tf.variable_scope(None, 'Openpose'):
            prefix = 'MConv_Stage1'
            (self.feed(feature_lv)
             # .se_block(name=prefix + '_L1_se', ratio=8)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L1_4')
             .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

            (self.feed(feature_lv)
             # .se_block(name=prefix + '_L2_se', ratio=8)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L2_4')
             .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            for stage_id in range(5):
                prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
                prefix = 'MConv_Stage%d' % (stage_id + 2)
                (self.feed(prefix_prev + '_L1_5',
                           prefix_prev + '_L2_5',
                           feature_lv)
                 .concat(3, name=prefix + '_concat')
                 # .se_block(name=prefix + '_L1_se', ratio=8)
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L1_4')
                 .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

                (self.feed(prefix + '_concat')
                 # .se_block(name=prefix + '_L2_se', ratio=8)
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L2_4')
                 .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            # final result
            (self.feed('MConv_Stage6_L2_5',
                       'MConv_Stage6_L1_5')
             .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def loss_last(self):
        return self.get_output('MConv_Stage6_L1_5'), self.get_output('MConv_Stage6_L2_5')

    def restorable_variables(self, only_backbone=True):
        vs = {v.op.name: v for v in tf.global_variables() if
              ('efficientnet-b0' in v.op.name or 'resample_p6' in v.op.name or 'fpn_cells' in v.op.name or (only_backbone is False and 'Openpose' in v.op.name)) and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'quant' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        # print(set([v.op.name for v in tf.global_variables()]) - set(list(vs.keys())))
        return vs

if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=(1, 384, 384, 3), name='image')
    network = EfficientdetNetwork({'image': input})

    print(network.restorable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['Openpose/concat_stage7']
                )

        with tf.gfile.GFile('../models/graph/efficientdet-d0/graph_opt.pb', 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)

