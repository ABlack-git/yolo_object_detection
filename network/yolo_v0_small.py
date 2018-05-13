from network.yolo_v0 import YoloV0
from network.yolo_v0_1 import YoloV01
import tensorflow as tf


class YoloSmall(YoloV0):

    def __init__(self, grid_size, img_size, params, restore=False):
        super(YoloSmall, self).__init__(grid_size, img_size, params, restore=restore)

    def inference(self, x):
        if not self.predictions:
            act_param = {'type': 'leaky', 'param': 0.1, 'write_summary': True}
            conv1 = super().create_conv_layer(x, [4, 4, 3, 16], 'Conv_1', [1, 2, 2, 1], activation=True, pooling=True,
                                              act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv2 = super().create_conv_layer(conv1, [4, 4, 16, 32], 'Conv_2', [1, 2, 2, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv3 = super().create_conv_layer(conv2, [3, 3, 32, 64], 'Conv_3', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv4 = super().create_conv_layer(conv3, [3, 3, 64, 128], 'Conv_4', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv5 = super().create_conv_layer(conv4, [3, 3, 128, 256], 'Conv_5', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv6 = super().create_conv_layer(conv5, [3, 3, 256, 128], 'Conv_6', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            in_dim = 6 * 4 * 128
            flatten = tf.reshape(conv6, [-1, in_dim])
            out_dim = self.grid_size[0] * self.grid_size[1] * 6 * self.no_boxes
            self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=False,
                                                       act_param={'type': 'sigmoid', 'write_summary': False},
                                                       weight_init='Xavier',
                                                       batch_norm=False)


class YoloSmallPretrain(YoloV01):
    def __init__(self, grid_size, img_size, params, restore=False):
        super(YoloSmallPretrain, self).__init__(grid_size, img_size, params, restore=restore)

    def inference(self, x):
        if not self.predictions:
            act_param = {'type': 'leaky', 'param': 0.1, 'write_summary': True}
            conv1 = super().create_conv_layer(x, [4, 4, 3, 16], 'Conv_1', [1, 2, 2, 1], activation=True, pooling=True,
                                              act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv2 = super().create_conv_layer(conv1, [4, 4, 16, 32], 'Conv_2', [1, 2, 2, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv3 = super().create_conv_layer(conv2, [3, 3, 32, 64], 'Conv_3', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv4 = super().create_conv_layer(conv3, [3, 3, 64, 128], 'Conv_4', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv5 = super().create_conv_layer(conv4, [3, 3, 128, 256], 'Conv_5', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True,
                                              trainable=True)

            conv6 = super().create_conv_layer(conv5, [3, 3, 256, 128], 'Conv_6', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)
            in_dim = 6 * 4 * 128
            flatten = tf.reshape(conv6, [-1, in_dim])
            out_dim = self.grid_size[0] * self.grid_size[1]
            self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=False,
                                                       act_param={'type': 'sigmoid', 'write_summary': False},
                                                       weight_init='Xavier',
                                                       batch_norm=False)