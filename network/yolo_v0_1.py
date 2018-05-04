
from network.yolo_v0 import YoloV0
import tensorflow as tf


class YoloV01(YoloV0):

    def __init__(self, grid_size, img_size, params, restore=False):
        super(YoloV01, self).__init__(grid_size, img_size, params, restore=restore)

    def loss_func(self, y_pred, y_true):
        if not self.loss:
            with tf.name_scope('Loss_function'):
                y_pred = tf.reshape(y_pred, [-1, self.grid_size[0] * self.grid_size[1], 5], name='reshape_pred')
                y_true = tf.reshape(y_true, [-1, self.grid_size[0] * self.grid_size[1], 5], name='reshape_truth')
                # define name scopes
                with tf.variable_scope('is_obj'):
                    is_obj = y_true[:, :, 4]
                with tf.variable_scope('t_x'):
                    t_x = y_true[:, :, 0]
                with tf.variable_scope('t_y'):
                    t_y = y_true[:, :, 1]
                with tf.variable_scope('t_w'):
                    t_w = y_true[:, :, 2]
                with tf.variable_scope('t_h'):
                    t_h = y_true[:, :, 3]
                with tf.variable_scope('p_x'):
                    p_x = y_pred[:, :, 0]
                with tf.variable_scope('p_y'):
                    p_y = y_pred[:, :, 1]
                with tf.variable_scope('p_w'):
                    p_w = y_pred[:, :, 2]
                with tf.variable_scope('p_h'):
                    p_h = y_pred[:, :, 3]
                with tf.variable_scope('p_c'):
                    p_c = y_pred[:, :, 4]

                with tf.name_scope('XY_LOSS'):
                    xy_loss = self.ph_coord_scale * tf.reduce_sum(
                        tf.multiply(is_obj, tf.pow(p_x - t_x, 2) + tf.pow(p_y - t_y, 2)), name='xy_loss')
                with tf.name_scope('WH_LOSS'):
                    wh_loss = self.ph_coord_scale * tf.reduce_sum(
                        tf.multiply(is_obj, tf.pow(p_w - t_w, 2) + tf.pow(p_h - t_h, 2)), name='wh_loss')

                # calculate confidance
                # iou = self.tf_iou(y_true, y_pred)
                # tf.stop_gradient(iou)
                # self.summary_list.append(tf.summary.histogram('IOU', iou))
                with tf.name_scope('Confidence'):
                    # confidence = tf.multiply(is_obj, iou, name='confidence')
                    # tf.stop_gradient(confidence)
                    with tf.variable_scope('no_obj'):
                        no_obj = tf.to_float(tf.not_equal(is_obj, 1))
                    # add confidence where object is present
                    with tf.variable_scope('obj_loss'):
                        c_obj_loss = self.ph_isobj_scale * tf.reduce_sum(
                            tf.multiply(is_obj, tf.pow(p_c - 1, 2)), name='c_obj_loss')
                    # add confidence where object is not present
                    with tf.variable_scope('noobj_loss'):
                        c_noobj_loss = self.ph_noobj_scale * tf.reduce_sum(tf.multiply(no_obj, tf.pow(p_c - 0, 2)),
                                                                           name='c_noobj_loss')
                with tf.variable_scope('Loss'):
                    self.loss = tf.add(xy_loss + wh_loss, c_obj_loss + c_noobj_loss, name='loss')

                self.summary_list.append(tf.summary.scalar('xy_loss', xy_loss))
                self.summary_list.append(tf.summary.scalar('wh_loss', wh_loss))
                self.summary_list.append(tf.summary.scalar('c_obj_loss', c_obj_loss))
                self.summary_list.append(tf.summary.scalar('c_noobj_loss', c_noobj_loss))
                self.summary_list.append(tf.summary.scalar('Loss', self.loss))
                self.summary_list.append(tf.summary.histogram('is_obj', is_obj))
                self.summary_list.append(tf.summary.histogram('no_obj', no_obj))
                # self.summary_list.append(tf.summary.histogram('confidance', confidence))
