import tensorflow as tf
from network.nn_template import NNTemplate


class YOLO(NNTemplate):

    def __init__(self, img_w, img_h, grid_w, grid_h, no_boxes):
        self.img_size = (img_w, img_h)
        self.grid_size = (grid_w, grid_h)
        self.no_boxes = no_boxes
        self.batch_size = None
        self.predictions = None
        self.loss = None
        self.optimizer = None

    def inference(self, x):
        if not self.predictions:
            act_param = {'type': 'leaky', 'param': 0.1}
            conv1 = super().create_conv_layer(x, [3, 3, 3, 16], 'Conv_1', [1, 1, 1, 1], activation=True, pooling=True,
                                              act_param=act_param)
            conv2 = super().create_conv_layer(conv1, [3, 3, 16, 32], 'Conv_2', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param)
            conv3 = super().create_conv_layer(conv2, [3, 3, 32, 64], 'Conv_3', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param)
            conv4 = super().create_conv_layer(conv3, [3, 3, 64, 128], 'Conv_4', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param)
            conv5 = super().create_conv_layer(conv4, [3, 3, 128, 256], 'Conv_5', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param)
            conv6 = super().create_conv_layer(conv5, [3, 3, 256, 512], 'Conv_6', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param)
            conv7 = super().create_conv_layer(conv6, [3, 3, 512, 1024], 'Conv_7', [1, 1, 1, 1], activation=True,
                                              pooling=False, act_param=act_param)
            conv8 = super().create_conv_layer(conv7, [3, 3, 1024, 256], 'Conv_8', [1, 1, 1, 1], activation=True,
                                              pooling=False, act_param=act_param)
            flatten = tf.reshape(conv8, [-1, 3 * 2 * 256])
            out_dim = self.grid_size[0] * self.grid_size[1] * 2 * self.no_boxes
            in_dim = 3 * 2 * 256
            self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=False)

    def loss(self, y_pred, y_true):
        """
        Create loss function
        :param y_pred: is Tensor of size [batch ,grid_w*grid_h*2*no_boxes*5]
        :param y_true: is Tensor of size [batch,grid_w*grid_h*2*no_boxes*5]
        :return:
        """
        if not self.loss:
            with tf.name_scope('Loss'):
                # reshape y_pred and y_truth to be [batch_size, grid_w*grid_h,B,5]
                y_pred = tf.reshape(y_pred, [self.batch_size, self.grid_size[0] * self.grid_size[1], self.no_boxes, 5])
                y_true = tf.reshape(y_true, [self.batch_size, self.grid_size[0] * self.grid_size[1], self.no_boxes, 5])
                # prpare masks
                is_object = tf.zeros([self.batch_size, self.grid_size[0] * self.grid_size[1], self.no_boxes])
                no_object = tf.zeros([self.batch_size, self.grid_size[0] * self.grid_size[1], self.no_boxes])
                # compute IoU
                iou = self.tf_iou(y_true, y_pred)
                pass

    def tf_iou(self, y_true, y_pred):
        """
        Computes IoU with tf functions.

        :param y_true: A tensor of size [batch_size, S, B, size>=4].
        :param y_pred: A tensor of size [batch_size, S, B, size>=4].
        Where S is the number of cells, B number of boxes and
        4 is the size of last dimension, that corresponds to x_c, y_c, w, h.
        :return: A tensor of size [batch_size, S, B]. Each celement of SxB matrix contains IoU for box B in cell S.
        """
        with tf.name_scope('IoU'):
            # convert center oriented coords to (x,y) of left and right corner
            x_tl_1 = y_true[:, :, :, 0] - tf.round(y_true[:, :, :, 2] / 2)
            y_tl_1 = y_true[:, :, :, 1] - tf.round(y_true[:, :, :, 3] / 2)
            x_br_1 = y_true[:, :, :, 0] + tf.round(y_true[:, :, :, 2] / 2)
            y_br_1 = y_true[:, :, :, 1] + tf.round(y_true[:, :, :, 3] / 2)

            x_tl_2 = y_pred[:, :, :, 0] - tf.round(y_pred[:, :, :, 2] / 2)
            y_tl_2 = y_pred[:, :, :, 1] - tf.round(y_pred[:, :, :, 3] / 2)
            x_br_2 = y_pred[:, :, :, 0] + tf.round(y_pred[:, :, :, 2] / 2)
            y_br_2 = y_pred[:, :, :, 1] + tf.round(y_pred[:, :, :, 3] / 2)

            # Compute coordinates of intersection box
            x_tl = tf.maximum(x_tl_1, x_tl_2)
            y_tl = tf.maximum(y_tl_1, y_tl_2)
            x_br = tf.minimum(x_br_1, x_br_2)
            y_br = tf.minimum(y_br_1, y_br_2)
            # Compute intersection area
            intersection = tf.multiply(tf.maximum(x_br - x_tl + 1, 0), tf.maximum(y_br - y_tl + 1, 0),
                                       name='Intersection')
            # Compute union area
            area_1 = (x_br_1 - x_tl_1 + 1) * (y_br_1 - y_tl_1 + 1)
            area_b = (x_br_2 - x_tl_2 + 1) * (y_br_2 - y_tl_2 + 1)
            union = tf.subtract(tf.add(area_1, area_b), intersection, name='Union')

        return tf.truediv(intersection, union, name='IoU')

    def optimize(self, epochs):
        raise NotImplementedError

    def save(self, sess, path, name):
        raise NotImplementedError

    def restore(self, sess, path, meta):
        raise NotImplementedError

    def iou(self, box_a, box_b):
        raise NotImplementedError
