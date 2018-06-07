import tensorflow as tf
import numpy as np
from network.nn_template import ANN
import configparser


class YOLO(ANN):

    def __init__(self, cfg):
        super(YOLO, self).__init__(cfg)

    def init_network(self, x, cfg):
        pass


    def loss_func(self, y_pred, y_true):
        raise NotImplementedError

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

    def nms(self, boxes):
        raise NotImplementedError

    def get_predictions(self, x):
        raise NotImplementedError

    def optimize(self, epochs):
        raise NotImplementedError

    def save(self, sess, path, name):
        raise NotImplementedError

    def restore(self, sess, path, meta):
        raise NotImplementedError
