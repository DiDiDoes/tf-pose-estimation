from __future__ import absolute_import

import tensorflow as tf
import numpy as np

import network_base
from efficientnet import efficientnet_builder
from network_base import layer

import tensorflow.contrib.slim as slim


class EfficientnetNetwork(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=1.0):
        self.conv_width = conv_width
        self.refine_width = conv_width2
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    @layer
    def base(self, input, name):
        net, endpoints = efficientnet_builder.build_model_base(input, name, training=True)
        for k, tensor in sorted(list(endpoints.items()), key=lambda x: x[0]):
            self.layers['%s/%s' % (name, k)] = tensor
            print(k, tensor.shape)
        return net

    def setup(self):
        depth2 = lambda x: int(x * self.refine_width)

        self.feed('image').base(name='efficientnet-b0')

        # for n, l in enumerate(self.layers):
        #     print(l)

        self.feed('efficientnet-b0/block_2').max_pool(2, 2, 2, 2, name='efficientnet-b0/block_2_pool')
        self.feed('efficientnet-b0/block_10').upsample(factor='efficientnet-b0/block_4', name='efficientnet-b0/block_10_upsample')
        (self.feed(
            'efficientnet-b0/block_2_pool',
            'efficientnet-b0/block_4',
            'efficientnet-b0/block_10_upsample',
            # 'base/layer_4/output/downsample'
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
              ('efficientnet-b0' in v.op.name or (only_backbone is False and 'Openpose' in v.op.name)) and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'quant' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        # print(set([v.op.name for v in tf.global_variables()]) - set(list(vs.keys())))
        print(len(tf.global_variables()))
        print(len(vs))
        print(len([v.op.name for v in tf.global_variables() if 'Openpose' in v.op.name]))
        return vs

if __name__ == '__main__':
    input1 = tf.placeholder(tf.float32, shape=(1, 384, 384, 3), name='image')
    input2 = tf.placeholder(tf.float32, shape=(1, 384, 384, 3), name='image')

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        network1 = EfficientnetNetwork({'image': input1})

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print(num_params)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        network2 = EfficientnetNetwork({'image': input2})

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print(num_params)
    network2.restorable_variables()
    #all_var = [t.name for t in tf.global_variables()]
    #print(all_var)
    glo = [v.op.name for v in tf.global_variables()]
    print(len(tf.global_variables()))
    trainable = [v.op.name for v in tf.trainable_variables()]
    print(len(tf.trainable_variables()))
    #print(set(glo) - set(trainable))
    print(len([v.op.name for v in tf.global_variables() if 'moving' in v.op.name]))
    l1s, l2s = network2.loss_l1_l2()
    print(len(l1s))
    l1l, l2l = network2.loss_last()
    print(l1l)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.cosine_decay(0.001, global_step, 100, alpha=0.0)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    var_list = [var for var in tf.trainable_variables() if var.name.startswith("Openpose")]
    print(var_list)
