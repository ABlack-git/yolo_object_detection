import tensorflow as tf
import math


class ANN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.summary_list = []
        self.layers_list = {}
        self.predictions = None
        self.loss = None
        self.global_step = None
        self.x = None
        self.y_true = None
        self.ph_train = None
        self.train = True

    def init_network(self, cfg):
        raise NotImplementedError

    def loss_func(self, predictions, labels):
        raise NotImplementedError

    def optimize(self, train_set, valid_set, num_epochs, param_dict):
        raise NotImplementedError

    def save(self, path, name):
        raise NotImplementedError

    def restore(self, path, meta=None, var_list=None):
        raise NotImplementedError

    def create_conv_layer(self, x, w_shape, name, strides=None, weight_init='Normal', batch_norm=True, trainable=True):
        if strides is None:
            strides = [1, 1, 1, 1]
        with tf.variable_scope(name):
            if weight_init == 'Normal':
                weights = tf.truncated_normal(w_shape, stddev=0.1)
            elif weight_init == 'Xavier':
                weights = tf.contrib.layers.xavier_initializer()
                weights = weights(w_shape)
            else:
                weights = tf.truncated_normal(w_shape, stddev=0.1)
            w = tf.Variable(weights, name='weights', trainable=trainable)
            conv = tf.nn.conv2d(x, w, strides=strides, padding='SAME', data_format='NHWC')
            self.summary_list.append(tf.summary.histogram('weights', w))
            if batch_norm:
                out = tf.layers.batch_normalization(conv, training=self.ph_train, name='batch_norm_layer',
                                                    trainable=trainable)
            else:
                b = tf.Variable(tf.constant(0.1, shape=[w_shape[3]]), name='biases', trainable=trainable)
                self.summary_list.append(tf.summary.histogram('biases', b))
                out = tf.add(conv, b, name='conv_and_bias')
            out = tf.identity(out, name='output')
        tf.logging.info('Layer %s created with parameters: ' % name)
        tf.logging.info('   weights shape: %s' % str(w_shape))
        tf.logging.info('   weights init: %s' % weight_init)
        tf.logging.info('   strides: %s' % str(strides))
        tf.logging.info('   trainable: %s' % trainable)
        tf.logging.info('   batch_norm: %s' % batch_norm)
        return out

    def create_fc_layer(self, x, shape, name, weight_init='Normal', batch_norm=True, trainable=True):
        with tf.variable_scope(name):
            if weight_init == 'Normal':
                weights = tf.truncated_normal(shape, stddev=0.1)
            elif weight_init == 'Xavier':
                weights = tf.contrib.layers.xavier_initializer()
                weights = weights(shape)
            else:
                weights = tf.truncated_normal(shape, stddev=0.1)

            w = tf.Variable(weights, name="weights", trainable=trainable)
            self.summary_list.append(tf.summary.histogram("weights", w))

            if batch_norm:
                out = tf.layers.batch_normalization(tf.matmul(x, w), training=self.ph_train, name='Batch_norm_layer',
                                                    trainable=trainable)
            else:
                b = tf.Variable(tf.constant(0.1, shape=[shape[1]]), name="biases", trainable=trainable)
                self.summary_list.append(tf.summary.histogram("biases", b))
                out = tf.add(tf.matmul(x, w), b, name='weighted_sum')
            out = tf.identity(out, name='output')
            self.summary_list.append(tf.summary.histogram('Output', out))
            tf.logging.info('Layer %s created with parameters: ' % name)
            tf.logging.info('   weights shape: %s' % str(shape))
            tf.logging.info('   weights init: %s' % weight_init)
            tf.logging.info('   trainable: %s' % trainable)
            tf.logging.info('   batch_norm: %s' % batch_norm)
            return out

    def create_activation_layer(self, x, act_type, params, name, write_summary):
        with tf.name_scope(name):
            if act_type == 'ReLU':
                activation = tf.nn.relu(x, name='relu')
            elif act_type == 'leaky':
                if params.get('alpha') is None:
                    raise ValueError('Parameter alpha is not specified in name scope %s' % name)
                activation = tf.nn.leaky_relu(x, float(params.get('alpha')), name='leaky_relu')
            elif act_type == 'sigmoid':
                activation = tf.sigmoid(x, name='sigmoid')
            else:
                raise ValueError('Unknown activation type %s in name scope %s' % (act_type, name))
            if write_summary:
                self.summary_list.append(tf.summary.histogram(name, activation))
            tf.logging.info('Activation function %s created with parameters: ' % name)
            tf.logging.info('   type: %s' % act_type)
            if params.get('alpha') is not None:
                tf.logging.info('   alpha: %s' % params.get('alpha'))
            tf.logging.info('   write summary: %s' % write_summary)
            return activation

    def create_pooling_layer(self, x, pool_type, kernel, strides, padding, name, write_summary):
        with tf.name_scope(name):
            if pool_type == 'max':
                pooling = tf.nn.max_pool(x, kernel, strides, padding, name='max_pool')
            else:
                raise ValueError('Unknown pooling type %s in name scope %s' % (pool_type, name))
            if write_summary:
                self.summary_list.append(tf.summary.histogram(name, pooling))
            tf.logging.info('Pooling layer %s created with parameters: ' % name)
            tf.logging.info('   type: %s' % pool_type)
            tf.logging.info('   kernel: %s' % str(kernel))
            tf.logging.info('   strides: %s' % str(strides))
            tf.logging.info('   padding: %s' % padding)
            return pooling

    def create_dropout(self, x, rate, name):
        with tf.name_scope(name):
            out = tf.layers.dropout(x, rate, self.ph_train, name='Dropout')
        tf.logging.info('Layer %s created with parameters: ' % name)
        tf.logging.info('   rate={:.2f}'.format(rate))
        return out

    def learning_rate(self, lr, g_step, hp, lr_type, offset=0):
        lr_type = lr_type.replace(" ", "")
        if lr_type == 'const':
            return lr
        if lr_type == 'exp_rise':
            return lr * math.exp(hp * (g_step - offset))
        if lr_type == 'exp_decay':
            return lr * math.exp(-hp * (g_step - offset))
        else:
            raise ValueError('Unknown learning rate type')
