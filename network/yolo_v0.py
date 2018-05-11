from network.nn_template import ANN
from data.dataset_generator import DatasetGenerator
import tensorflow as tf
import os
import datetime
import numpy as np
import time


class YoloV0(ANN):

    def __init__(self, grid_size, img_size, params=None, restore=False):
        """
        :param grid_size: Size of grid
        :param img_size: Size of input image
        :param params: loss_scale, training_set_imgs, training_set_labels, batch_size, learning_rate, optimizer,
        threshold
        :param restore: Boolean. True if model should be restored from saved parameters. False if new model should be
        created
        """
        self.params = params
        self.set_logger_verbosity()
        # Graph elements and tensorflow functions
        super(YoloV0, self).__init__()
        self.optimizer = None
        self.sess = None
        self.saver = None
        self.ph_learning_rate = None
        self.ph_train = None
        self.ph_coord_scale = None
        self.ph_noobj_scale = None
        self.ph_isobj_scale = None
        # Model parameters
        if params is None:
            params = {'coord_scale': 5,
                      'noobj_scale': 0.5,
                      'isobj_scale': 1,
                      'training_set_imgs': None,
                      'training_set_labels': None,
                      'testing_set_imgs': None,
                      'testing_set_labels': None,
                      'batch_size': 1,
                      'learning_rate': 0.01,
                      'optimizer': 'SGD',
                      'opt_para': None,
                      'threshold': 0.3,
                      'save_path': 'CheckPoints'}
        self.restored = False
        self.no_boxes = 1
        self.grid_size = grid_size
        self.img_size = img_size
        self.coord_scale = params.get('coord_scale')
        self.noobj_scale = params.get('noobj_scale')
        self.isobj_scale = params.get('isobj_scale')
        self.batch_size = params.get('batch_size')
        self.learning_rate = params.get('learning_rate')
        self.nms_threshold = params.get('threshold')
        # Data sets
        if params.get('training_set_imgs') is not None:
            self.training_set = DatasetGenerator(params.get('training_set_imgs'), params.get('training_set_labels'),
                                                 self.img_size, grid_size, 1)
        self.valid_set = None
        if params.get('testing_set_imgs') is not None:
            self.test_set = DatasetGenerator(params.get('testing_set_imgs'), params.get('testing_set_labels'),
                                             self.img_size,
                                             grid_size, 1)
        self.save_path = params.get('save_path')
        # Model initialization
        self.open_sess()

        if not restore:
            self.__create_network(params)
        else:
            pass

    def __create_network(self, params):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.x = tf.placeholder(tf.float32, [None, self.img_size[1], self.img_size[0], 3], name='Input')
        self.y_true = tf.placeholder(tf.float32, [None, self.grid_size[0] * self.grid_size[1] * 5], name='GT_input')
        self.ph_learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.ph_coord_scale = tf.placeholder(tf.float32, shape=(), name='coord_scale')
        self.ph_noobj_scale = tf.placeholder(tf.float32, shape=(), name='noobj_scale')
        self.ph_isobj_scale = tf.placeholder(tf.float32, shape=(), name='isobj_scale')
        self.ph_train = tf.placeholder(tf.bool, name='training')
        self.inference(self.x)
        self.loss_func(self.predictions, self.y_true)
        self._optimizer(params.get('optimizer'), params.get('opt_param'), write_grads=False)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)

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
                iou = self.tf_iou(y_true, y_pred)
                tf.stop_gradient(iou)
                self.summary_list.append(tf.summary.histogram('IOU', iou))
                with tf.name_scope('Confidence'):
                    confidence = tf.multiply(is_obj, iou, name='confidence')
                    tf.stop_gradient(confidence)
                    with tf.variable_scope('no_obj'):
                        no_obj = tf.to_float(tf.not_equal(is_obj, 1))
                    # add confidence where object is present
                    with tf.variable_scope('obj_loss'):
                        c_obj_loss = self.ph_isobj_scale * tf.reduce_sum(
                            tf.multiply(is_obj, tf.pow(p_c - confidence, 2)), name='c_obj_loss')
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
                self.summary_list.append(tf.summary.histogram('confidance', confidence))

    def inference(self, x):
        if not self.predictions:
            act_param = {'type': 'leaky', 'param': 0.1, 'write_summary': False}
            conv1 = super().create_conv_layer(x, [3, 3, 3, 16], 'Conv_1', [1, 1, 1, 1], activation=True, pooling=True,
                                              act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv2 = super().create_conv_layer(conv1, [3, 3, 16, 32], 'Conv_2', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv3 = super().create_conv_layer(conv2, [3, 3, 32, 64], 'Conv_3', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv4 = super().create_conv_layer(conv3, [3, 3, 64, 128], 'Conv_4', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv5 = super().create_conv_layer(conv4, [3, 3, 128, 256], 'Conv_5', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv6 = super().create_conv_layer(conv5, [3, 3, 256, 512], 'Conv_6', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv7 = super().create_conv_layer(conv6, [3, 3, 512, 1024], 'Conv_7', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)

            conv8 = super().create_conv_layer(conv7, [3, 3, 1024, 256], 'Conv_8', [1, 1, 1, 1], activation=True,
                                              pooling=True, act_param=act_param, weight_init='Xavier', batch_norm=True)
            # 3x2 is size of a feature map in last conv layer
            flatten = tf.reshape(conv8, [-1, 3 * 2 * 256])
            out_dim = self.grid_size[0] * self.grid_size[1] * 5 * self.no_boxes
            in_dim = 3 * 2 * 256
            self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=False,
                                                       act_param={'type': 'sigmoid', 'write_summary': False},
                                                       weight_init='Xavier',
                                                       batch_norm=True)

    def _optimizer(self, optimizer='Adam', param=None, write_grads=True):
        if not self.optimizer:
            with tf.name_scope('Optimizer'):
                if optimizer == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(self.ph_learning_rate)
                    tf.logging.info('Using %s optimizer' % optimizer)
                elif optimizer == 'SGD':
                    tf.logging.info('Using %s optimizer' % optimizer)
                    self.optimizer = tf.train.GradientDescentOptimizer(self.ph_learning_rate)
                elif optimizer == 'AdaGrad':
                    tf.logging.info('Using %s optimizer' % optimizer)
                    self.optimizer = tf.train.AdagradOptimizer(self.ph_learning_rate)
                elif optimizer == 'Nesterov':
                    tf.logging.info('Using %s optimizer' % optimizer)
                    self.optimizer = tf.train.MomentumOptimizer(self.ph_learning_rate, param, use_nesterov=True)
                else:
                    tf.logging.warning('Optimizer specified in input is not supported. Exiting.')
                    exit(1)
                # add summaries of gradients
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = self.optimizer.compute_gradients(self.loss)
                self.optimizer = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='optimizer')
                if write_grads:
                    for i, grad in enumerate(grads):
                        self.summary_list.append(
                            tf.summary.histogram("{}-grad".format(grads[i][1].name.replace(':0', '-0')), grads[i]))

    def optimize(self, epochs, sum_path):
        now = datetime.datetime.now()
        summary_folder = '%d_%d_%d__%d-%d' % (now.day, now.month, now.year, now.hour, now.minute)
        summary_writer = tf.summary.FileWriter(os.path.join(sum_path, summary_folder), graph=tf.get_default_graph())
        summary = tf.summary.merge_all()
        tf.logging.info(
            'Starting to train model. Current global step is %s' % tf.train.global_step(self.sess, self.global_step))
        for _ in range(epochs):
            batch = self.training_set.get_minibatch(self.batch_size)
            no_batches = self.training_set.get_number_of_batches(self.batch_size)
            for i in range(no_batches):
                t_0 = time.time()
                imgs, labels = next(batch)
                g_step = tf.train.global_step(self.sess, self.global_step)
                if g_step % 50 == 0:
                    val_t0 = time.time()
                    s = self.sess.run(summary, feed_dict={self.x: imgs, self.y_true: labels,
                                                          self.ph_learning_rate: self.learning_rate,
                                                          self.ph_coord_scale: self.coord_scale,
                                                          self.ph_noobj_scale: self.noobj_scale,
                                                          self.ph_train: False,
                                                          self.ph_isobj_scale: self.isobj_scale})
                    summary_writer.add_summary(s, tf.train.global_step(self.sess, self.global_step))
                    summary_writer.flush()
                    loss, preds = self.sess.run([self.loss, self.predictions],
                                                feed_dict={self.x: imgs, self.y_true: labels,
                                                           self.ph_learning_rate: self.learning_rate,
                                                           self.ph_coord_scale: self.coord_scale,
                                                           self.ph_noobj_scale: self.noobj_scale,
                                                           self.ph_train: False,
                                                           self.ph_isobj_scale: self.isobj_scale})
                    print(np.asarray(preds, np.float32))
                    b_preds = self.predictions_to_boxes(preds)
                    b_true = self.predictions_to_boxes(labels)
                    b_true = self.convert_coords(b_true)
                    tmp = []
                    for b_img in b_true:
                        tmp.append(np.delete(b_img, np.where(b_img[:, 4] != 1.0), axis=0))
                    b_true = tmp
                    b_preds = self.nms(b_preds)
                    stats = self.compute_stats(b_preds, b_true)
                    tmp = np.sum(stats, axis=0)
                    no_tp = tmp[0]
                    avg_prec = tmp[1] / len(stats)
                    avg_recall = tmp[2] / len(stats)
                    avg_conf = tmp[3] / len(stats)
                    avg_iou = tmp[4] / len(stats)
                    self.log_scalar('t_avg_prec', avg_prec, summary_writer, 'Statistics')
                    self.log_scalar('t_avg_recall', avg_recall, summary_writer, 'Statistics')
                    self.log_scalar('t_avg_conf', avg_conf, summary_writer, 'Statistics')
                    self.log_scalar('t_avg_iou', avg_iou, summary_writer, 'Statistics')
                    val_tf = time.time() - val_t0
                    print('Statistics on training set')
                    print(
                        'Step: %s, loss: %.4f, no_tp: %d, avg_precision: %.3f, avg_recall %.3f, avg_confidance: %.3f, '
                        'avg_iou: %.3f, Valiadation time: %.2f'
                        % (tf.train.global_step(self.sess, self.global_step), loss, no_tp, avg_prec, avg_recall,
                           avg_conf, avg_iou, val_tf))
                if i % 200 == 0:
                    self.test_model()

                # if i == int(self.training_set.get_number_of_batches(self.batch_size) / 2):
                #     self.save(self.save_path, 'model')

                self.sess.run([self.optimizer],
                              feed_dict={self.x: imgs, self.y_true: labels, self.ph_learning_rate: self.learning_rate,
                                         self.ph_coord_scale: self.coord_scale,
                                         self.ph_noobj_scale: self.noobj_scale,
                                         self.ph_train: True,
                                         self.ph_isobj_scale: self.isobj_scale})
                t_f = time.time() - t_0
                tf.logging.info('Global step: %s, Batch processed: %d/%d, Time to process batch: %.2f' % (
                    tf.train.global_step(self.sess, self.global_step), i + 1, no_batches, t_f))
            # save every epoch
            self.save(self.save_path, 'model')

    def tf_iou(self, y_true, y_pred):
        """
        Computes IoU with tf functions.

        :param y_true: A tensor of size [batch_size, S*S, size>=4].
        :param y_pred: A tensor of size [batch_size, S*S, size>=4].
        Where S is the number of cells, B number of boxes and
        4 is the size of last dimension, that corresponds to x_c, y_c, w, h.
        :return: A tensor of size [batch_size, S], where each element is IoU of predicted box with ground truth box.
        """
        with tf.name_scope('IoU'):
            Gi = np.zeros([self.grid_size[0] * self.grid_size[1]])
            Gj = np.zeros([self.grid_size[0] * self.grid_size[1]])
            counter_i = 0
            counter_j = 0
            for i in range(self.grid_size[0] * self.grid_size[1]):
                Gi[i] = counter_i
                Gj[i] = counter_j
                counter_i += 1
                if (i + 1) % self.grid_size[0] == 0:
                    counter_i = 0
                    counter_j += 1

            with tf.name_scope('Resize'):
                g_i = tf.constant(Gi, name='row_offset', dtype=tf.float32)
                g_j = tf.constant(Gj, name='column_offset', dtype=tf.float32)

                y_tx = (y_true[:, :, 0] + g_i) * self.img_size[0] / self.grid_size[0]
                y_ty = (y_true[:, :, 1] + g_j) * self.img_size[1] / self.grid_size[1]
                y_tw = tf.pow(y_true[:, :, 2] * self.img_size[0], 2)
                y_th = tf.pow(y_true[:, :, 3] * self.img_size[1], 2)

                y_px = (y_pred[:, :, 0] + g_i) * self.img_size[0] / self.grid_size[0]
                y_py = (y_pred[:, :, 1] + g_j) * self.img_size[1] / self.grid_size[1]
                y_pw = tf.pow(y_pred[:, :, 2] * self.img_size[0], 2)
                y_ph = tf.pow(y_pred[:, :, 3] * self.img_size[1], 2)
            with tf.name_scope('Covert_coords'):
                x_tl_1 = y_tx - tf.round(y_tw / 2)
                y_tl_1 = y_ty - tf.round(y_th / 2)
                x_br_1 = y_tx + tf.round(y_tw / 2)
                y_br_1 = y_ty + tf.round(y_th / 2)

                x_tl_2 = y_px - tf.round(y_pw / 2)
                y_tl_2 = y_py - tf.round(y_ph / 2)
                x_br_2 = y_px + tf.round(y_pw / 2)
                y_br_2 = y_py + tf.round(y_ph / 2)
            with tf.name_scope('Compute_IOU'):
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

    def log_scalar(self, tag, value, summary_writer, name):
        with tf.name_scope(name):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        summary_writer.add_summary(summary, tf.train.global_step(self.sess, self.global_step))

    def get_predictions(self, x):
        if x is None:
            raise TypeError
        return self.sess.run(self.predictions, feed_dict={self.x: x, self.ph_train: False})

    def predictions_to_boxes(self, preds):
        """
        Coverts predictions of network to bounding box format.
        :param preds: Predictions of the network. Shape [batch_size,S*S*5].
        :return: Returns ALL boxes predicted by the network. Boxes coordinates corespond to pixels.
        Shape of returned tensor is [batch_size, S*S, 5]
        """
        preds = np.reshape(preds, [self.batch_size, self.grid_size[0] * self.grid_size[1], 5])
        counter_i = 0
        counter_j = 0
        for i in range(self.grid_size[0] * self.grid_size[1]):
            preds[:, i, 0] = np.round((preds[:, i, 0] + counter_i) * self.img_size[0] / self.grid_size[0])
            preds[:, i, 1] = np.round((preds[:, i, 1] + counter_j) * self.img_size[1] / self.grid_size[1])
            preds[:, i, 2] = np.round(np.power(preds[:, i, 2] * self.img_size[0], 2))
            preds[:, i, 3] = np.round(np.power(preds[:, i, 3] * self.img_size[1], 2))
            counter_i += 1
            if (i + 1) % self.grid_size[0] == 0:
                counter_i = 0
                counter_j += 1
        return preds

    def nms(self, preds):
        """
        Non-maximum suppression algorithm
        :param preds: Boxes predicted by the network(Boxes coords are in corners). Shape: [batch_size, S*S, 5]
        :return: List of np.arrays of shape [batch_size, ?, 5], that correspond to best boxes in batches
        """
        preds = self.convert_coords(preds)
        # sort boxes by their confidence
        indeces = np.argsort(preds[:, :, 4])
        picked = []
        for batch_n, batch_ind in enumerate(indeces):
            batch_picked = []
            # delete indexies with confidence < threshold
            batch_i = np.delete(batch_ind, np.where(preds[batch_n, batch_ind, 4] <= self.nms_threshold))
            while len(batch_i) > 0:
                last = len(batch_i) - 1
                # pick box with highest confidence
                batch_picked.append(preds[batch_n, batch_i[last], :])
                # calculate IoU of all boxes with picked box
                picked_box = np.tile(preds[batch_n, batch_i[last], :], [len(batch_i), 1])
                ious = self.iou(picked_box, preds[batch_n, batch_i, :])
                # delete indexies that have IoU>=threshold
                batch_i = np.delete(batch_i, np.where(ious >= 0.5))
            picked.append(np.array(batch_picked))
        return picked

    def iou(self, boxes_a, boxes_b):
        """
        Computes IoU.
        Requires coordinates in x_tl, y_tl, x_br, y_br format
        :param boxes_a: Tensor of shape [S*S, 5]
        :param boxes_b: Tensor of shape [S*S, 5]
        :return: IoU of boxes_a and boxes_b
        """
        x_a = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
        y_a = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
        x_b = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
        y_b = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

        inter_area = np.multiply(np.maximum((x_b - x_a + 1), 0), np.maximum((y_b - y_a + 1), 0))
        a_area = np.multiply(boxes_a[:, 2] - boxes_a[:, 0] + 1, boxes_a[:, 3] - boxes_a[:, 1] + 1)
        b_area = np.multiply(boxes_b[:, 2] - boxes_b[:, 0] + 1, boxes_b[:, 3] - boxes_b[:, 1] + 1)
        union_area = a_area + b_area - inter_area
        return np.divide(inter_area, union_area)

    def convert_coords(self, boxes, nump=True):
        """
        Converts center based coordinates to coordinates of two corners
        :param boxes: Array of boxes with shape [bacth_size, S*S, 5]
        :return: Boxes with converted coordinates
        """
        if nump:
            boxes[:, :, 0] = boxes[:, :, 0] - np.round(boxes[:, :, 2] / 2)
            boxes[:, :, 1] = boxes[:, :, 1] - np.round(boxes[:, :, 3] / 2)
            boxes[:, :, 2] = boxes[:, :, 2] + boxes[:, :, 0]
            boxes[:, :, 3] = boxes[:, :, 3] + boxes[:, :, 1]
            return boxes
        else:
            boxes_ret = []
            for item_n, batch_item in enumerate(boxes):
                batch_item[:, 0] = batch_item[:, 0] - np.round(batch_item[:, 2] / 2)
                batch_item[:, 1] = batch_item[:, 1] - np.round(batch_item[:, 3] / 2)
                batch_item[:, 2] = batch_item[:, 2] + batch_item[:, 0]
                batch_item[:, 3] = batch_item[:, 3] + batch_item[:, 1]
                boxes_ret.append(batch_item)
            return boxes_ret

    def set_logger_verbosity(self, verbosity=tf.logging.INFO):
        tf.logging.set_verbosity(verbosity)

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, os.path.join(path, name),
                        global_step=tf.train.global_step(self.sess, self.global_step))

    def restore(self, path, meta=None, var_names=None):
        if not self.restored:
            if meta is not None:
                self.saver = tf.train.import_meta_graph(meta)
                self.saver.restore(self.sess, save_path=path)
            elif var_names is not None:
                self.__create_network(self.params)
                # # [var for var in tf.global_variables() if var.op.name=="bar"][0]
                # variables = []
                # for var_name in var_names:
                #     for var in tf.global_variables():
                #         if var.op.name == var_name:
                #             variables.append(var)
                graph = tf.get_default_graph()
                variables = [graph.get_tensor_by_name(name=name + ':0') for name in var_names]
                saver = tf.train.Saver(variables)
                saver.restore(self.sess, save_path=path)
            else:
                tf.logging.info('Restore pass was not specified, exiting.')
                exit(1)
            try:

                graph = tf.get_default_graph()
                self.x = graph.get_tensor_by_name('Input:0')
                self.y_true = graph.get_tensor_by_name('GT_input:0')
                self.predictions = graph.get_tensor_by_name('FC_1/Sigmoid:0')
                self.loss = graph.get_tensor_by_name('Loss_function/Loss/loss:0')
                self.optimizer = graph.get_operation_by_name('Optimizer/optimizer')
                self.global_step = graph.get_tensor_by_name('global_step:0')
                self.ph_train = graph.get_tensor_by_name('training:0')
                self.ph_learning_rate = graph.get_tensor_by_name('learning_rate:0')
                self.ph_noobj_scale = graph.get_tensor_by_name('noobj_scale:0')
                self.ph_coord_scale = graph.get_tensor_by_name('coord_scale:0')
                self.ph_isobj_scale = graph.get_tensor_by_name('isobj_scale:0')
                # self.saver = tf.train.Saver(max_to_keep=10)
                self.restored = True
            except KeyError as e:
                tf.logging.fatal("Restoring was not successful. KeyError exception was raised.")
                tf.logging.fatal(e)
                exit(1)

    def open_sess(self):
        if not self.sess:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # writer = tf.summary.FileWriter(os.path.join('summaries', 'graph'), graph=tf.get_default_graph(),
            #                                filename_suffix='graph_')
            # writer.flush()

    def close_sess(self):
        self.sess.close()
        self.sess = None

    def compute_stats(self, pred_boxes, true_boxes):
        """
        Computes number of correctly detected objects and their average confidence and average iou with ground truth.
        Also computes precision and recall. This is all done for every image in the bacth.
        :param pred_boxes: List of np.arrays of shape [batch_size, ?, 5]
        :param true_boxes: List of np.arrays [batch_size, ?, 5]
        :return return lis of size [batch_size, 5]. [no_tp, precision, recall, avg_conf, avg_iou]
        """
        statistics = []
        for img_num, p_boxes in enumerate(pred_boxes):
            batch_ious = np.zeros([len(p_boxes), len(true_boxes[img_num])])
            for box_n, box in enumerate(p_boxes):
                box = np.tile(box, [len(true_boxes[img_num]), 1])
                batch_ious[box_n, :] = self.iou(box, true_boxes[img_num])

            # true positives
            tp = [-1 for _ in range(np.shape(batch_ious)[0])]
            # find  all true positives
            for t in range(len(true_boxes[img_num])):
                max_iou = -1
                p_assign = -1
                for p in range(np.shape(batch_ious)[0]):
                    if batch_ious[p, t] >= 0.5:
                        if tp[p] == -1:
                            if batch_ious[p, t] > max_iou:
                                max_iou = batch_ious[p, t]
                                p_assign = p
                if p_assign != -1:
                    tp[p_assign] = t
            # false negatives
            fn = [fn for fn in range(len(true_boxes[img_num])) if fn not in tp]
            # false positives
            fp = [fp for fp, ind in enumerate(tp) if ind == -1]

            no_fn = float(len(fn))
            no_fp = float(len(fp))
            no_tp = 0.0
            avg_iou = 0.0
            avg_conf = 0.0
            for ind, element in enumerate(tp):
                if element != -1:
                    avg_iou += batch_ious[ind, element]
                    avg_conf += p_boxes[ind, 4]
                    no_tp += 1.0

            avg_iou = avg_iou / no_tp if no_tp > 0 else 0.0
            avg_conf = avg_conf / no_tp if no_tp > 0 else 0.0
            precision = no_tp / (no_tp + no_fp) if (no_tp + no_fp) > 0 else 0.0
            recall = no_tp / (no_tp + no_fn) if (no_tp + no_fn) > 0 else 0.0
            statistics.append([no_tp, precision, recall, avg_conf, avg_iou])

        return np.asarray(statistics)

    def test_model(self, batch_size):
        raise NotImplementedError


if __name__ == '__main__':
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size)
    net.close_sess()
