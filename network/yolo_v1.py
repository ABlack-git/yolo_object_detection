from network.yolo_template import YOLO
import tensorflow as tf


class YoloV1(YOLO):

    def __init__(self, img_w, img_h, grid_w, grid_h, no_boxes, scale_1, scale_2):
        self.img_size = (img_w, img_h)
        self.grid_size = (grid_w, grid_h)
        self.no_boxes = no_boxes
        self.batch_size = None
        self.predictions = None
        self.loss = None
        self.optimizer = None
        self.corrd_scale = scale_1
        self.noobj_scale = scale_2

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
            out_dim = self.grid_size[0] * self.grid_size[1] * 5 * self.no_boxes
            in_dim = 3 * 2 * 256
            self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=True,
                                                       act_param={'type': 'sigmoid'})

    def loss_func(self, y_pred, y_true):
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
                y_true = tf.reshape(y_true, [self.batch_size, self.grid_size[0] * self.grid_size[1], self.no_boxes, 6])
                # BTW, maybe I can apply sigmoid func to the output of the network, to keep values between 0-1.
                # Though problems with vanishing grad can occur.

                # compute IoU
                iou = self.tf_iou(y_true, y_pred)

                # C=P(obj)*IOU(truth, pred). This will compute confidence for every box.
                confidence = tf.multiply(y_true[:, :, :, 4], iou)

                # Create masks
                # this (maybe) returns Batch x S*S x B tensor
                is_obj = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=2, keepdims=True)))
                # elements in y_true[:,:,:,5] are all set to 1 if there is object in cell S_i.
                is_obj = tf.multiply(is_obj, y_true[:, :, :, 5])

                no_obj = tf.to_float(tf.not_equal(is_obj, 1))

                # Loss
                xy_loss = self.corrd_scale * tf.reduce_sum(
                    is_obj * (tf.pow(y_pred[:, :, :, 0] - y_true[:, :, :, 0], 2) +
                              tf.pow(y_pred[:, :, :, 1] - y_true[:, :, :, 1], 2)))
                wh_loss = self.corrd_scale * tf.reduce_sum(
                    is_obj * (tf.pow(y_pred[:, :, :, 2] - y_true[:, :, :, 2], 2) +
                              tf.pow(y_pred[:, :, :, 3] - y_true[:, :, :, 3], 2)))
                c_obj_loss = tf.reduce_sum(
                    is_obj * (tf.pow(y_pred[:, :, :, 4] - confidence, 2)))
                c_noobj_loss = self.noobj_scale * tf.reduce_sum(
                    no_obj * (tf.pow(y_pred[:, :, :, 4] - confidence, 2)))

                tf.summary.scalar('Coordinates loss', xy_loss)
                tf.summary.scalar('Width-height loss', wh_loss)
                tf.summary.scalar('Object confidence loss', c_obj_loss)
                tf.summary.scalar('No object confidence loss', c_noobj_loss)

                self.loss = xy_loss + wh_loss + c_obj_loss + c_noobj_loss
                tf.summary.scalar('Total loss', self.loss)
