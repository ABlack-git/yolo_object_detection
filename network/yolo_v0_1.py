from network.yolo_v0 import YoloV0
import tensorflow as tf
import datetime
import os
import time
import numpy as np


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
                # with tf.variable_scope('t_x'):
                #     t_x = y_true[:, :, 0]
                # with tf.variable_scope('t_y'):
                #     t_y = y_true[:, :, 1]
                # with tf.variable_scope('t_w'):
                #     t_w = y_true[:, :, 2]
                # with tf.variable_scope('t_h'):
                #     t_h = y_true[:, :, 3]
                # with tf.variable_scope('p_x'):
                #     p_x = y_pred[:, :, 0]
                # with tf.variable_scope('p_y'):
                #     p_y = y_pred[:, :, 1]
                # with tf.variable_scope('p_w'):
                #     p_w = y_pred[:, :, 2]
                # with tf.variable_scope('p_h'):
                #     p_h = y_pred[:, :, 3]
                # with tf.variable_scope('p_c'):
                #     p_c = y_pred[:, :, 4]

                loss = tf.reduce_sum(tf.pow(y_pred - is_obj, 2))
                self.summary_list.append(tf.summary.scalar('Loss', loss))
                self.loss = loss

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
                                                          self.ph_train: False})
                    summary_writer.add_summary(s, tf.train.global_step(self.sess, self.global_step))
                    summary_writer.flush()
                    loss, preds = self.sess.run([self.loss, self.predictions],
                                                feed_dict={self.x: imgs, self.y_true: labels,
                                                           self.ph_learning_rate: self.learning_rate,
                                                           self.ph_train: False})
                    # Compute accuracy
                    preds = np.asarray(preds)
                    preds[np.where(preds >= 0.6)] = 1
                    preds[np.where(preds < 0.6)] = 0
                    labels = np.reshape(labels,
                                        [self.batch_size, self.grid_size[0] * self.grid_size[1], 5])
                    tp = (preds == labels[:, 4]).astype(dtype=int)
                    accuracy = np.sum(tp) / (self.grid_size[0] * self.grid_size[1] * self.batch_size)
                    self.log_scalar('Accuracy', accuracy, summary_writer, 'Statistics')
                    val_tf = time.time() - val_t0
                    tf.logging.info('Statistics on training set')
                    tf.logging.info('Step: %s, loss: %f.4, accuracy: %f.3, time: %f.3' % (
                        tf.train.global_step(self.sess, self.global_step), loss, accuracy, val_tf))

                if (g_step + 1) % 100 == 0:
                    self.test_model()

                self.sess.run([self.optimizer],
                              feed_dict={self.x: imgs, self.y_true: labels,
                                         self.ph_learning_rate: self.learning_rate,
                                         self.ph_train: True})
                t_f = time.time() - t_0
                tf.logging.info('Global step: %s, Batch processed: %d/%d, Time to process batch: %.2f' % (
                    tf.train.global_step(self.sess, self.global_step), i + 1, no_batches, t_f))
                # save every epoch
            self.save(self.save_path, 'model')
