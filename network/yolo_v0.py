from network.nn_template import ANN
from data.dataset_generator import DatasetGenerator
import tensorflow as tf
import os
import datetime
import numpy as np
import time
import configparser
import json
import stats_utils as su
import bbox_utils as bbu


class YoloV0(ANN):

    def __init__(self, cfg):
        self.set_logger_verbosity()
        # Graph elements and tensorflow functions
        super(YoloV0, self).__init__(cfg)
        self.optimizer = None
        self.sess = None
        self.saver = None
        self.ph_learning_rate = None
        self.ph_train = None
        self.ph_noobj_scale = None
        self.ph_isobj_scale = None
        self.ph_prob_noobj = None
        self.ph_prob_isobj = None
        self.ph_xy_scale = None
        self.ph_wh_scale = None
        self.ph_weight_decay = None
        # Model parameters
        self.grid_size = None
        self.img_size = None
        self.epoch_step = None
        self.xy_scale = None
        self.wh_scale = None
        self.noobj_scale = None
        self.isobj_scale = None
        self.prob_noobj = None
        self.prob_isobj = None
        self.batch_size = None
        self.learning_rate = None
        self.nms_threshold = None
        self.no_boxes = 1
        self.optimizer_param = None
        self.lr_policy = None
        self.lr_param = None
        self.outputs_per_box = -1
        self.weight_decay = -1
        # strings
        self.optimizer_type = ''
        self.model_version = ''
        # bools
        self.restored = False
        self.write_grads = False
        self.sqrt = True
        self.keep_asp_ratio = False
        # Model initialization
        self.open_sess()
        self.__create_network()

    def __create_network(self):
        self.ph_learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.ph_wh_scale = tf.placeholder(tf.float32, shape=(), name='wh_scale')
        self.ph_xy_scale = tf.placeholder(tf.float32, shape=(), name='xy_scale')
        self.ph_noobj_scale = tf.placeholder(tf.float32, shape=(), name='noobj_scale')
        self.ph_isobj_scale = tf.placeholder(tf.float32, shape=(), name='isobj_scale')
        self.ph_train = tf.placeholder(tf.bool, name='training')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.init_network(self.cfg)
        if self.outputs_per_box > 5:
            self.ph_prob_noobj = tf.placeholder(tf.float32, shape=(), name='prob_noobj')
            self.ph_prob_isobj = tf.placeholder(tf.float32, shape=(), name='prob_noobj')
        if self.weight_decay > 0:
            self.ph_weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        self.loss_func(self.predictions, self.y_true)
        self._optimizer(self.optimizer_type, self.optimizer_param, write_summary=self.write_grads)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=30)

    def loss_func(self, y_pred, y_true):
        if not self.loss:
            with tf.name_scope('Loss_function'):
                y_pred = tf.reshape(y_pred, [-1, self.grid_size[0] * self.grid_size[1], self.outputs_per_box],
                                    name='reshape_pred')
                y_true = tf.reshape(y_true, [-1, self.grid_size[0] * self.grid_size[1], 5], name='reshape_truth')
                # define name scopes for better representation in tensorboard
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
                    xy_loss = self.ph_xy_scale * tf.reduce_sum(
                        tf.multiply(is_obj, tf.pow(p_x - t_x, 2) + tf.pow(p_y - t_y, 2)), name='xy_loss')
                with tf.name_scope('WH_LOSS'):
                    wh_loss = self.ph_wh_scale * tf.reduce_sum(
                        tf.multiply(is_obj, tf.pow(p_w - t_w, 2) + tf.pow(p_h - t_h, 2)), name='wh_loss')

                # calculate confidance
                iou = self.tf_iou(y_true, y_pred)
                tf.stop_gradient(iou)
                self.summary_list.append(tf.summary.histogram('IOU', iou))
                with tf.variable_scope('no_obj'):
                    no_obj = tf.to_float(tf.not_equal(is_obj, 1))
                # add confidence where object is present
                with tf.variable_scope('obj_loss'):
                    c_obj_loss = self.ph_isobj_scale * tf.reduce_sum(
                        tf.multiply(is_obj, tf.pow(p_c - iou, 2)), name='c_obj_loss')
                # add confidence where object is not present
                with tf.variable_scope('noobj_loss'):
                    c_noobj_loss = self.ph_noobj_scale * tf.reduce_sum(tf.multiply(no_obj, tf.pow(p_c - iou, 2)),
                                                                       name='c_noobj_loss')

                if self.outputs_per_box > 5:
                    with tf.variable_scope('p_prob'):
                        p_prob = y_pred[:, :, 5]
                    with tf.variable_scope('prob_obj_loss'):
                        prob_obj_loss = self.ph_prob_isobj * tf.reduce_sum(
                            tf.multiply(is_obj, tf.pow(p_prob - is_obj, 2)),
                            name='prob_obj_loss')
                    with tf.variable_scope('prob_noobj_loss'):
                        prob_noobj_loss = self.ph_prob_noobj * tf.reduce_sum(
                            tf.multiply(no_obj, tf.pow(p_prob - is_obj, 2)),
                            name='prob_noobj_loss')
                    self.summary_list.append(tf.summary.scalar('prob_obj_loss', prob_obj_loss))
                    self.summary_list.append(tf.summary.scalar('prob_noobj_loss', prob_noobj_loss))
                    with tf.variable_scope('Loss'):
                        self.loss = tf.add_n([xy_loss, wh_loss, c_obj_loss, c_noobj_loss, prob_obj_loss,
                                              prob_noobj_loss], name='loss')

                else:
                    self.loss = tf.add_n([xy_loss, wh_loss, c_obj_loss, c_noobj_loss], name='loss')
                if self.weight_decay > 0:
                    with tf.variable_scope('l2_loss'):
                        l2_loss = self.ph_weight_decay * tf.add_n([tf.nn.l2_loss(w, name='L2_loss')
                                                                   for w in tf.trainable_variables()])
                    self.loss = self.loss + l2_loss
                    self.summary_list.append(tf.summary.scalar('l2_loss', l2_loss))

                self.summary_list.append(tf.summary.scalar('xy_loss', xy_loss))
                self.summary_list.append(tf.summary.scalar('wh_loss', wh_loss))
                self.summary_list.append(tf.summary.scalar('c_obj_loss', c_obj_loss))
                self.summary_list.append(tf.summary.scalar('c_noobj_loss', c_noobj_loss))
                self.summary_list.append(tf.summary.scalar('Loss', self.loss))
                self.summary_list.append(tf.summary.histogram('is_obj', is_obj))
                self.summary_list.append(tf.summary.histogram('no_obj', no_obj))
        return self.loss

    def init_network(self, cfg):
        parser = configparser.ConfigParser()
        parser.read(cfg)
        last_out = None
        for section in parser.sections():
            if section == 'PARAMETERS':
                # mandatory parameters
                self.grid_size = [int(val) for val in parser.get(section, 'grid_size').split(',')]
                self.img_size = [int(val) for val in parser.get(section, 'img_size').split(',')]
                self.x = tf.placeholder(tf.float32, [None, self.img_size[1], self.img_size[0], 3], name='Input')
                last_out = self.x
                self.y_true = tf.placeholder(tf.float32, [None, self.grid_size[0] * self.grid_size[1] * 5],
                                             name='labels')
                self.epoch_step = [int(val) for val in parser.get(section, 'epoch_step').split(',')]
                self.learning_rate = [float(val) for val in parser.get(section, 'learning_rate').split(',')]
                self.xy_scale = [float(val) for val in parser.get(section, 'xy_scale').split(',')]
                self.wh_scale = [float(val) for val in parser.get(section, 'wh_scale').split(',')]
                self.noobj_scale = [float(val) for val in parser.get(section, 'noobj_scale').split(',')]
                self.isobj_scale = [float(val) for val in parser.get(section, 'isobj_scale').split(',')]

                if len(self.learning_rate) != len(self.epoch_step):
                    raise ValueError('Length of learning rate array is not equal to epoch step array')
                if (len(self.xy_scale) != len(self.epoch_step)) or (len(self.wh_scale) != len(self.epoch_step)):
                    raise ValueError('Length of xy or wh scale arrays is not equal to epoch step array')
                if len(self.noobj_scale) != len(self.epoch_step):
                    raise ValueError('Length of noobj scale array is not equal to epoch step array')
                if len(self.isobj_scale) != len(self.epoch_step):
                    raise ValueError('Length of isobj scale array is not equal to epoch step array')

                self.nms_threshold = parser.getfloat(section, 'nms_threshold')
                self.batch_size = parser.getint(section, 'batch_size')
                self.model_version = parser.get(section, 'model_version')
                self.optimizer_type = parser.get(section, 'optimizer')
                # optional
                if parser.has_option(section, 'weight_decay'):
                    self.weight_decay = parser.getfloat(section, 'weight_decay')
                if parser.has_option(section, 'lr_policy'):
                    self.lr_policy = [val for val in parser.get(section, 'lr_policy').split(',')]
                    if len(self.lr_policy) != len(self.epoch_step):
                        raise ValueError('Length of lr_policy array is not equal to epoch step array')
                else:
                    self.lr_policy = ['const' for _ in range(len(self.learning_rate))]

                if parser.has_option(section, 'lr_param'):
                    self.lr_param = [float(val) for val in parser.get(section, 'lr_param').split(',')]
                    if len(self.lr_param) != len(self.epoch_step):
                        raise ValueError('Length of lr_param array is not equal to epoch step array')
                else:
                    self.lr_param = [1 for _ in range(len(self.epoch_step))]

                if parser.has_option(section, 'no_boxes'):
                    self.no_boxes = parser.getint(section, 'no_boxes')
                else:
                    self.no_boxes = 1
                if parser.has_option(section, 'optimizer_param'):
                    self.optimizer_param = parser.getfloat(section, 'optimizer_param')
                if parser.has_option(section, 'sqrt'):
                    self.sqrt = parser.getboolean(section, 'sqrt')
                else:
                    self.sqrt = True

                if parser.has_option(section, 'write_grads'):
                    self.write_grads = parser.getboolean(section, 'write_grads')
                else:
                    self.write_grads = False
                if parser.has_option(section, 'outputs_per_box'):
                    self.outputs_per_box = parser.getint(section, 'outputs_per_box')
                    if self.outputs_per_box > 5:
                        self.prob_noobj = [float(val) for val in parser.get(section, 'prob_noobj').split(',')]
                        self.prob_isobj = [float(val) for val in parser.get(section, 'prob_isobj').split(',')]
                        if len(self.prob_noobj) != len(self.epoch_step):
                            raise ValueError('Length of prob noobj array is not equal to epoch step array')
                        if len(self.prob_isobj) != len(self.epoch_step):
                            raise ValueError('Length of prob isobj array is not equal to epoch step array')
                else:
                    self.outputs_per_box = 5

                if parser.has_option(section, 'keep_asp_ratio'):
                    self.keep_asp_ratio = parser.getboolean(section, 'keep_asp_ratio')

            elif section.startswith('CONV'):
                name = section
                w_shape = [int(val) for val in parser.get(section, 'w_shape').split(',')]
                batch_norm = parser.getboolean(section, 'batch_norm')
                weight_init = parser.get(section, 'weight_init')
                if parser.has_option(section, 'strides'):
                    strides = [int(val) for val in parser.get(section, 'strides').split(',')]
                else:
                    strides = None
                if parser.has_option(section, 'trainable'):
                    trainable = parser.getboolean(section, 'trainable')
                else:
                    trainable = True
                self.layers_list[name] = super().create_conv_layer(last_out, w_shape, name, strides=strides,
                                                                   weight_init=weight_init, batch_norm=batch_norm,
                                                                   trainable=trainable)
                last_out = self.layers_list[name]
            elif section.startswith('ACTIVATION'):
                name = parser.get(section, 'name')
                act_type = parser.get(section, 'type')
                if parser.has_option(section, 'write_summary'):
                    write_summary = parser.getboolean(section, 'write_summary')
                else:
                    write_summary = False
                params = {}
                for option, value in parser.items(section):
                    if option != 'name' and option != 'type' and option != 'name':
                        params[option] = value
                self.layers_list[name] = super().create_activation_layer(last_out, act_type, params, name,
                                                                         write_summary)
                last_out = self.layers_list[name]
            elif section.startswith('POOLING'):
                name = parser.get(section, 'name')
                pool_type = parser.get(section, 'type')
                kernel_size = [int(val) for val in parser.get(section, 'kernel_size').split(',')]
                strides = [int(val) for val in parser.get(section, 'strides').split(',')]
                padding = parser.get(section, 'padding')
                if parser.has_option(section, 'write_summary'):
                    write_summary = parser.getboolean(section, 'write_summary')
                else:
                    write_summary = False
                self.layers_list[name] = super().create_pooling_layer(last_out, pool_type, kernel_size, strides,
                                                                      padding, name, write_summary)
                last_out = self.layers_list[name]

            elif section.startswith('DROPOUT'):
                name = section
                rate = parser.getfloat(section, 'rate')
                mode = None
                if parser.has_option(section, 'mode'):
                    mode = parser.get(section, 'mode')
                if mode is not None:
                    self.layers_list[name] = super().create_dropout(last_out, rate, name, mode)
                else:
                    self.layers_list[name] = super().create_dropout(last_out, rate, name)
                last_out = self.layers_list[name]

            elif section.startswith('FC'):
                name = section
                w_shape = [int(val) for val in parser.get(section, 'w_shape').split(',')]
                batch_norm = parser.getboolean(section, 'batch_norm')
                weight_init = parser.get(section, 'weight_init')
                if parser.has_option(section, 'trainable'):
                    trainable = parser.getboolean(section, 'trainable')
                else:
                    trainable = True
                if parser.has_option(section, 'reshape'):
                    reshape = parser.getint(section, 'reshape')
                    last_out = tf.reshape(last_out, [-1, reshape])
                self.layers_list[name] = super().create_fc_layer(last_out, w_shape, name, weight_init=weight_init,
                                                                 batch_norm=batch_norm, trainable=trainable)
                last_out = self.layers_list[name]
            else:
                raise ValueError('Unknown section %s in the configuration file' % section)
        self.predictions = last_out
        return self.predictions

    def _optimizer(self, optimizer='Adam', param=None, write_summary=True):
        if not self.optimizer:
            with tf.name_scope('Optimizer'):
                if optimizer == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(self.ph_learning_rate, epsilon=0.1)
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
                elif optimizer == 'Momentum':
                    self.optimizer = tf.train.MomentumOptimizer(self.ph_learning_rate, param, use_nesterov=False)
                else:
                    tf.logging.warning('Optimizer specified in input is not supported. Exiting.')
                    exit(1)
                # add summaries of gradients
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = self.optimizer.compute_gradients(self.loss)
                self.optimizer = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='optimizer')
                if write_summary:
                    for i, grad in enumerate(grads):
                        self.summary_list.append(
                            tf.summary.histogram("{}-grad".format(grads[i][1].name.replace(':0', '-0')), grads[i]))

    def optimize(self, train_set, valid_set, no_epochs, param_dict):
        # init additional parameters
        summ_step = param_dict['summary_step']
        do_test = param_dict['do_test']
        start_step = tf.train.global_step(self.sess, self.global_step)
        # set save and summary folders
        now = datetime.datetime.now()
        model_folder = os.path.join(param_dict['summary_path'], self.model_version)
        summary_folder = '%d_%d_%d__%d-%d' % (now.day, now.month, now.year, now.hour, now.minute)
        summary_writer = tf.summary.FileWriter(os.path.join(model_folder, summary_folder), graph=tf.get_default_graph())
        summary = tf.summary.merge_all()
        save_path = os.path.join(param_dict['save_path'], self.model_version)
        # init datasets
        ts_cfg = {"images": train_set['images'], 'annotations': train_set['labels'],
                  'configuration': {"img_size": {'width': self.img_size[0], 'height': self.img_size[1]},
                                    'grid_size': {'width': self.grid_size[0], 'height': self.grid_size[1]},
                                    'no_boxes': self.no_boxes,
                                    'shuffle': True, 'sqrt': self.sqrt, 'keep_asp_ratio': self.keep_asp_ratio}}
        train_set = DatasetGenerator(json.dumps(ts_cfg))
        train_test_set = None
        if do_test:
            if valid_set is not None:
                vs_cfg = {"images": valid_set['images'], 'annotations': valid_set['labels'],
                          'configuration': {"img_size": {'width': self.img_size[0], 'height': self.img_size[1]},
                                            'grid_size': {'width': self.grid_size[0], 'height': self.grid_size[1]},
                                            'no_boxes': self.no_boxes,
                                            'shuffle': True, 'sqrt': self.sqrt, 'keep_asp_ratio': self.keep_asp_ratio}}
                valid_set = DatasetGenerator(json.dumps(vs_cfg))
                ts_cfg['configuration']['subset_length'] = valid_set.get_dataset_size()
            else:
                ts_cfg['configuration']['subset_length'] = int(train_set.get_dataset_size() / 10)
            train_test_set = DatasetGenerator(json.dumps(ts_cfg))

        # start training
        tf.logging.info('Starting to train model. Current global step is %s'
                        % tf.train.global_step(self.sess, self.global_step))
        for _ in range(no_epochs):
            batch = train_set.get_minibatch(self.batch_size)
            no_batches = train_set.get_number_of_batches(self.batch_size)
            for i in range(no_batches):
                t_0 = time.time()
                imgs, labels = next(batch)
                g_step = tf.train.global_step(self.sess, self.global_step)
                # compute index for lr schedule
                ind = 0
                for k, e_step in enumerate(self.epoch_step):
                    if (g_step / no_batches) < e_step:
                        ind = k
                        break
                    if k == len(self.epoch_step) - 1:
                        ind = k
                lr = super().learning_rate(self.learning_rate[ind], g_step, self.lr_param[ind], self.lr_policy[ind],
                                           start_step)
                # construct feed_dict
                feed_dict = {self.x: imgs, self.y_true: labels,
                             self.ph_learning_rate: lr,
                             self.ph_wh_scale: self.wh_scale[ind],
                             self.ph_xy_scale: self.xy_scale[ind],
                             self.ph_noobj_scale: self.noobj_scale[ind],
                             self.ph_train: True,
                             self.ph_isobj_scale: self.isobj_scale[ind]}

                if self.outputs_per_box > 5:
                    feed_dict.update({self.ph_prob_noobj: self.prob_noobj[ind],
                                      self.ph_prob_isobj: self.prob_isobj[ind]})
                if self.weight_decay > 0:
                    feed_dict.update({self.ph_weight_decay: self.weight_decay})
                # write summary
                if (g_step + 1) % summ_step == 0:
                    s = self.sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(s, tf.train.global_step(self.sess, self.global_step))
                    summary_writer.flush()
                # update weights
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict=feed_dict)
                t_f = time.time() - t_0
                # print updates
                epoch = int(g_step / no_batches) + 1
                tf.logging.info('Global step: %d, epoch: %d, loss: %.3f, Batch processed: %d/%d, '
                                'Time to process batch: %.2f' % (g_step, epoch, loss, i + 1, no_batches, t_f))
                if self.outputs_per_box > 5:
                    tf.logging.info('Learning rate %.2e, xy scale: %.2e, wh scale: %.2e, noobj scale: %.2e, is obj '
                                    'scale: %.2e, prob noobj: %.2e, prob is obj: %.2e' % (lr, self.xy_scale[ind],
                                                                                          self.wh_scale[ind],
                                                                                          self.noobj_scale[ind],
                                                                                          self.isobj_scale[ind],
                                                                                          self.prob_noobj[ind],
                                                                                          self.prob_isobj[ind]))
                else:
                    tf.logging.info('Learning rate %.2e, xy scale: %.2e, wh scale: %.2e, noobj scale: %.2e, is obj '
                                    'scale: %.2e' % (lr, self.xy_scale[ind],
                                                     self.wh_scale[ind],
                                                     self.noobj_scale[ind],
                                                     self.isobj_scale[ind]))

            # save every epoch
            self.save(save_path, self.model_version)
            if do_test and (valid_set is not None):
                val_stats = self.__validate_model(valid_set, 0.5, prefix='Validation model on validation set')
                self.log_scalar('Statistics', 'validation_avg_prec', val_stats[0], summary_writer, )
                self.log_scalar('Statistics', 'validation_avg_recall', val_stats[1], summary_writer)
                self.log_scalar('Statistics', 'validation_avg_iou', val_stats[2], summary_writer)
                self.log_scalar('Statistics', 'validation_avg_conf_tp', val_stats[3], summary_writer)
                self.log_scalar('Statistics', 'validation_avg_conf_fp', val_stats[4], summary_writer)
                self.log_scalar('Statistics', 'validation_num_of_tp', val_stats[5], summary_writer)
            if do_test:
                val_stats = self.__validate_model(train_test_set, 0.5, prefix='Validation model on subset of train set')
                self.log_scalar('Statistics', 'train_avg_prec', val_stats[0], summary_writer)
                self.log_scalar('Statistics', 'train_avg_recall', val_stats[1], summary_writer)
                self.log_scalar('Statistics', 'train_avg_iou', val_stats[2], summary_writer)
                self.log_scalar('Statistics', 'train_avg_conf_tp', val_stats[3], summary_writer)
                self.log_scalar('Statistics', 'train_avg_conf_fp', val_stats[4], summary_writer)
                self.log_scalar('Statistics', 'train_num_of_tp', val_stats[5], summary_writer)

    def tf_iou(self, y_true, y_pred):
        """
        Computes IoU using tensorflow functions. As an input this function takes tensors of predictions and ground
        truth. Data in those tensors is represented in grid cell format and internally converted to conventional
        bounding box format.

        :param y_true: A tensor of size [batch_size, S*S, size>=4].
        :param y_pred: A tensor of size [batch_size, S*S, size>=4].
        Where S is the number of cells.
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

                y_px = (y_pred[:, :, 0] + g_i) * self.img_size[0] / self.grid_size[0]
                y_py = (y_pred[:, :, 1] + g_j) * self.img_size[1] / self.grid_size[1]
                if self.sqrt:
                    y_pw = tf.pow(y_pred[:, :, 2], 2) * self.img_size[0]
                    y_ph = tf.pow(y_pred[:, :, 3], 2) * self.img_size[1]
                    y_tw = tf.pow(y_true[:, :, 2], 2) * self.img_size[0]
                    y_th = tf.pow(y_true[:, :, 3], 2) * self.img_size[1]
                else:
                    y_pw = y_pred[:, :, 2] * self.img_size[0]
                    y_ph = y_pred[:, :, 3] * self.img_size[1]
                    y_tw = y_true[:, :, 2] * self.img_size[0]
                    y_th = y_true[:, :, 3] * self.img_size[1]
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
                area_1 = tf.maximum((x_br_1 - x_tl_1 + 1) * (y_br_1 - y_tl_1 + 1), 1)
                area_b = tf.maximum((x_br_2 - x_tl_2 + 1) * (y_br_2 - y_tl_2 + 1), 1)
                union = tf.subtract(tf.add(area_1, area_b), intersection, name='Union')

                return tf.truediv(intersection, union, name='IoU')

    def log_scalar(self, scope_name, value_name, value, summary_writer):
        tag = scope_name + '/' + value_name
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        summary_writer.add_summary(summary, tf.train.global_step(self.sess, self.global_step))

    def get_predictions(self, x):
        if x is None:
            raise TypeError
        preds = self.sess.run(self.predictions, feed_dict={self.x: x, self.ph_train: False})
        preds = self.predictions_to_boxes(preds)
        preds = self.convert_coords(preds)
        # preds = bbu.convert_center_to_2points(preds)
        ret = []
        for img in preds:
            ret.append(self.non_max_suppression(img))
        return ret

    def predictions_to_boxes(self, in_preds):
        """
        Coverts predictions of network to bounding box format.
        :param in_preds: Predictions of the network. Shape [batch_size,S*S*5].
        :param last_dim_size: Size of last dimension.
        :return: Returns ALL boxes predicted by the network. Boxes coordinates corespond to pixels.
        Shape of returned tensor is [batch_size, S*S, 5]
        """
        preds = np.copy(in_preds)
        preds = np.reshape(preds, [len(preds), self.grid_size[0] * self.grid_size[1], self.outputs_per_box])
        counter_i = 0
        counter_j = 0
        for i in range(self.grid_size[0] * self.grid_size[1]):
            preds[:, i, 0] = np.round((preds[:, i, 0] + counter_i) * self.img_size[0] / self.grid_size[0])
            preds[:, i, 1] = np.round((preds[:, i, 1] + counter_j) * self.img_size[1] / self.grid_size[1])
            if self.sqrt:
                preds[:, i, 2] = np.round(np.power(preds[:, i, 2], 2) * self.img_size[0])
                preds[:, i, 3] = np.round(np.power(preds[:, i, 3], 2) * self.img_size[1])
            else:
                preds[:, i, 2] = np.round(preds[:, i, 2] * self.img_size[0])
                preds[:, i, 3] = np.round(preds[:, i, 3] * self.img_size[1])
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

    def non_max_suppression(self, preds):
        """
        Non-maximum suppression algorithm.
        :param preds: np array of boxes in form of top left and bottom right coordinates predicted by the network with
        shape [grid_w*grid_h, 5]
        :return: Boxes per image that correspond to predictions. Can return empty list.
        """
        indices = np.argsort(preds[:, 4])
        # delete all elements with low confidence
        indices = np.delete(indices, np.where(preds[indices, 4] <= self.nms_threshold))
        picked = []
        while len(indices) > 0:
            i = indices[len(indices) - 1]
            picked.append(i)
            indices = np.delete(indices, len(indices) - 1)
            current_box = np.tile(preds[i, :], (len(indices), 1))
            iou = self.iou(current_box, preds[indices, :])
            indices = np.delete(indices, np.where(iou >= 0.5))
        return preds[picked, :]

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
        union_area = np.maximum(a_area + b_area - inter_area, 1)
        return np.divide(inter_area, union_area)

    def convert_coords(self, in_boxes, nump=True):
        """
        Converts center based coordinates to coordinates of two corners
        :param in_boxes: Array of boxes with shape [bacth_size, S*S, 5]
        :return: Boxes with converted coordinates
        """

        if nump:
            boxes = np.copy(in_boxes)
            boxes[:, :, 0] = boxes[:, :, 0] - np.round(boxes[:, :, 2] / 2)
            boxes[:, :, 1] = boxes[:, :, 1] - np.round(boxes[:, :, 3] / 2)
            boxes[:, :, 2] = boxes[:, :, 2] + boxes[:, :, 0]
            boxes[:, :, 3] = boxes[:, :, 3] + boxes[:, :, 1]
            return boxes
        else:
            boxes_ret = []
            boxes = in_boxes[:, :, :]
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

    def graph_summary(self, summary_path):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        path = os.path.join(summary_path, self.model_version)
        writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
        writer.flush()

    def restore(self, path, meta=None, var_names=None):
        if not self.restored:
            if meta is not None:
                self.saver = tf.train.import_meta_graph(meta)
                self.saver.restore(self.sess, save_path=path)
                try:
                    graph = tf.get_default_graph()
                    self.x = graph.get_tensor_by_name('Input:0')
                    self.y_true = graph.get_tensor_by_name('labels:0')
                    self.predictions = graph.get_tensor_by_name('FC_1/output:0')
                    self.loss = graph.get_tensor_by_name('Loss_function/Loss/loss:0')
                    self.optimizer = graph.get_operation_by_name('Optimizer/optimizer')
                    self.global_step = graph.get_tensor_by_name('global_step:0')
                    self.ph_train = graph.get_tensor_by_name('training:0')
                    self.ph_learning_rate = graph.get_tensor_by_name('learning_rate:0')
                    self.ph_noobj_scale = graph.get_tensor_by_name('noobj_scale:0')
                    self.ph_xy_scale = graph.get_tensor_by_name('xy_scale:0')
                    self.ph_wh_scale = graph.get_tensor_by_name('wh_scale:0')
                    self.ph_isobj_scale = graph.get_tensor_by_name('isobj_scale:0')
                    # self.saver = tf.train.Saver(max_to_keep=10)
                    self.restored = True
                except KeyError as e:
                    tf.logging.fatal("Restoring was not successful. KeyError exception was raised.")
                    tf.logging.fatal(e)
                    exit(1)
            elif path is not None:
                # self.__create_network()
                variables = None
                if var_names is not None:
                    graph = tf.get_default_graph()
                    variables = [graph.get_tensor_by_name(name=name + ':0') for name in var_names]
                saver = tf.train.Saver(var_list=variables)
                saver.restore(self.sess, save_path=path)
            else:
                tf.logging.info('Restore pass was not specified, exiting.')
                exit(1)

    def open_sess(self):
        if not self.sess:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def close_sess(self):
        self.sess.close()
        self.sess = None

    def __validate_model(self, dataset: DatasetGenerator, iou_threshold, prefix=''):
        batch = dataset.get_minibatch(self.batch_size, resize_only=True)
        num_batches = dataset.get_number_of_batches(self.batch_size)
        stats = []

        for i in range(num_batches):
            img, labels = next(batch)
            true_boxes = []
            for batch_bb in labels:
                true_boxes.append(bbu.convert_center_to_2points(batch_bb))
            preds = self.get_predictions(img)
            stats = su.compute_stats(preds, true_boxes, iou_threshold, stats)
            su.progress_bar(i, num_batches, prefix=prefix)
        final_stats = su.process_stats(stats)
        print(prefix)
        print('Average precision: {0[0]:.3f}, Average recall: {0[1]:.3f}, Average iou: {0[2]:.3f}, '
              'Average confidence of TP: {0[3]:.3f}, '
              'Average confidence of FP: {0[4]:.3f}, Total num of TP: {0[5]}, Total num of FP: {0[6]}, '
              'Total num of FN: {0[7]}'.format(final_stats))
        return final_stats
