from network.yolo_v0 import YoloV0
import tensorflow as tf
import datetime
import os
import time
import numpy as np
import cv2

class YoloV01(YoloV0):

    def __init__(self, grid_size, img_size, params, restore=False):
        super(YoloV01, self).__init__(grid_size, img_size, params, restore=restore)

    def inference(self, x):
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
        in_dim = 3 * 2 * 256
        flatten = tf.reshape(conv8, [-1, in_dim])
        out_dim = self.grid_size[0] * self.grid_size[1]
        self.predictions = super().create_fc_layer(flatten, [in_dim, out_dim], 'FC_1', activation=True,
                                                   act_param={'type': 'sigmoid', 'write_summary': True},
                                                   weight_init='Xavier',
                                                   batch_norm=False)

    def loss_func(self, y_pred, y_true):
        if not self.loss:
            with tf.name_scope('Loss_function'):
                # y_pred = tf.reshape(y_pred, [-1, self.grid_size[0] * self.grid_size[1], 5], name='reshape_pred')
                y_true = tf.reshape(y_true, [-1, self.grid_size[0] * self.grid_size[1], 5], name='reshape_truth')
                # define name scopes
                with tf.variable_scope('is_obj'):
                    is_obj = y_true[:, :, 4]
                with tf.variable_scope('no_obj'):
                    no_obj = tf.to_float(tf.not_equal(is_obj, 1))

                isobj_loss = self.ph_isobj_scale * tf.reduce_sum(tf.multiply(is_obj, tf.pow(y_pred - 1, 2)),
                                                                 name='isobj_loss')
                noobj_loss = self.ph_noobj_scale * tf.reduce_sum(tf.multiply(no_obj, tf.pow(y_pred - 0, 2)),
                                                                 name='noobj_loss')
                self.summary_list.append(tf.summary.scalar('isobj_loss', isobj_loss))
                self.summary_list.append(tf.summary.scalar('noobj_loss', noobj_loss))
                self.loss = tf.add(isobj_loss, noobj_loss, name='total_loss')
                self.summary_list.append(tf.summary.scalar('total_loss', self.loss))

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
                                                          self.ph_train: False,
                                                          self.ph_isobj_scale: self.isobj_scale,
                                                          self.ph_noobj_scale: self.noobj_scale})
                    summary_writer.add_summary(s, tf.train.global_step(self.sess, self.global_step))
                    summary_writer.flush()
                    preds, loss = self.sess.run([self.predictions, self.loss],
                                                feed_dict={self.x: imgs, self.y_true: labels,
                                                           self.ph_learning_rate: self.learning_rate,
                                                           self.ph_train: False,
                                                           self.ph_isobj_scale: self.isobj_scale,
                                                           self.ph_noobj_scale: self.noobj_scale})
                    # Compute statistics
                    precision, recall, no_tp = self.compute_stats(preds, labels)
                    self.log_scalar('Precision', precision, summary_writer, 'Statistics')
                    self.log_scalar('Recall', recall, summary_writer, 'Statistics')
                    val_tf = time.time() - val_t0
                    tf.logging.info('Statistics on training set')
                    tf.logging.info('Step: %s, no_tp: %d, loss: %.2f, precision: %.2f, recall: %.2f, time: %.2f' % (
                        tf.train.global_step(self.sess, self.global_step), no_tp, loss, precision, recall, val_tf))

                if g_step % 200 == 0:
                    tf.logging.info('Statistics on testing set at step %s' % g_step)
                    self.test_model(self.batch_size, summary_writer)

                self.sess.run([self.optimizer],
                              feed_dict={self.x: imgs, self.y_true: labels,
                                         self.ph_learning_rate: self.learning_rate,
                                         self.ph_train: True,
                                         self.ph_isobj_scale: self.isobj_scale,
                                         self.ph_noobj_scale: self.noobj_scale})
                t_f = time.time() - t_0
                tf.logging.info('Global step: %s, Batch processed: %d/%d, Time to process batch: %.2f' % (
                    tf.train.global_step(self.sess, self.global_step), i + 1, no_batches, t_f))
                # save every epoch
            self.save(self.save_path, 'model')

    def compute_stats(self, preds, labels):
        preds = np.asarray(preds)
        preds[np.where(preds > 0.5)] = 1
        preds[np.where(preds <= 0.5)] = 0
        no_cells = self.grid_size[0] * self.grid_size[1]
        tmp_labels = np.reshape(labels,
                                [self.batch_size, no_cells, 5])
        no_tp = np.sum(preds[np.where(tmp_labels[:, :, 4] == 1)])
        preds[np.where(tmp_labels[:, :, 4] == 1)] = 0
        no_fp = np.sum(preds)
        no_fn = np.sum(
            np.equal(tmp_labels[:, :, 4], np.ones([self.batch_size, no_cells])).astype(int)) - no_tp
        precision = no_tp / (no_tp + no_fp) if (no_tp + no_fp) > 0 else 0
        recall = no_tp / (no_tp + no_fn) if (no_tp + no_fn) > 0 else 0
        return precision, recall, int(no_tp)

    def restore(self, path, meta=None, var_list=None):
        if not self.restored:
            self.saver = tf.train.import_meta_graph(meta)
            # self.saver = tf.train.Saver(max_to_keep=10)
            try:
                self.saver.restore(self.sess, save_path=path)
                graph = tf.get_default_graph()
                self.x = graph.get_tensor_by_name('Input:0')
                self.y_true = graph.get_tensor_by_name('GT_input:0')
                self.predictions = graph.get_tensor_by_name('FC_1/Sigmoid:0')
                self.loss = graph.get_tensor_by_name('Loss_function/total_loss:0')
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

    def test_model(self, batch_size, summary_writer=None):
        t_0 = time.time()
        batches = self.test_set.get_minibatch(batch_size)
        no_batches = self.test_set.get_number_of_batches(batch_size)
        precision = np.zeros(no_batches)
        recall = np.zeros(no_batches)
        for b in range(no_batches):
            imgs, labels = next(batches)
            preds = self.get_predictions(imgs)
            precision[b], recall[b], _ = self.compute_stats(preds, labels)
        avg_precision = np.sum(precision) / no_batches
        avg_recall = np.sum(recall) / no_batches
        t_f = time.time() - t_0
        if summary_writer is not None:
            self.log_scalar('Avg_precision', avg_precision, summary_writer, name='Statistics')
            self.log_scalar('Avg_recall', avg_recall, summary_writer, name='Statistics')
        tf.logging.info('Avg_precision: %.4f, avg_recall: %.4f, time: %.2f' % (avg_precision, avg_recall, t_f))

    def predictions_to_cells(self, preds):
        w_cells = self.grid_size[0]
        h_cells = self.grid_size[1]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        c_width = img_width / w_cells
        c_height = img_height / h_cells
        coords = []
        for batch in preds:
            for p_i, p in enumerate(batch):
                if p > 0.5:
                    x = (p_i % w_cells) * c_width + int(c_width / 2)
                    y = (np.floor(p_i / w_cells)) * c_height + int(c_height / 2)
                    coords.append([x, y])
        return coords

    def print_trainable_variables(self):
        for var in tf.trainable_variables:
            print(var)
