import tensorflow as tf


class ANN:
    def __init__(self):
        self.summary_list = []
        self.predictions = None
        self.loss = None
        self.global_step = None
        self.x = None
        self.y_true = None
        self.train = None

    def inference(self, x):
        raise NotImplementedError

    def loss_func(self, predictions, labels):
        raise NotImplementedError

    def optimize(self, epochs):
        raise NotImplementedError

    def save(self, path, name):
        raise NotImplementedError

    def restore(self, path, meta):
        raise NotImplementedError

    def create_conv_layer(self, x, w_shape, name, strides=None, activation=True, pooling=True, act_param=None,
                          pool_param=None, weight_init='Normal', batch_norm=True
                          ):

        """
        This function will create convolutional layer with activation function and pooling layer under same
        tensorflow name scope.
        :param x: A Tensor. Must be one of the following types: half, bfloat16, float32.
        :param w_shape: A Tensor. Must have the same type as input.
        A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        :param name: Name for tensorflow name scope.
        :param strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of
        input.
        :param activation: Boolean value. If true, it will create an activation function in this layer.
        :param pooling: Boolean value. If true, it will create a pooling layer in this layer.
        :param act_param: dict with keys 'type' and 'param'.
        :param pool_param: dict with keys 'type', 'kernel', 'strides', 'padding'.
        :param weight_init: Determines how to initialize weights. 'Normal' will init weights with normal distribution.
        :return: A Tensor. Has the same type as input.
        """

        if strides is None:
            strides = [1, 1, 1, 1]

        if activation and act_param is None:
            act_param = {'type': 'ReLU', 'param': None}

        if pooling and pool_param is None:
            pool_param = {'type': 'max', 'kernel': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'}

        # if weight_init != 'Normal':
        #     tf.logging.warning('Currently weight initialization is supported with normal distribution. '
        #                        'Continue with Normal distribution')

        with tf.name_scope(name):
            if weight_init == 'Normal':
                weights = tf.truncated_normal(w_shape, stddev=0.1)
            elif weight_init == 'Xavier':
                weights = tf.contrib.layers.xavier_initializer()
                weights = weights(w_shape)
                tf.logging.info('Using Xavier init for %s layer' % name)
            else:
                weights = tf.truncated_normal(w_shape, stddev=0.1)
            w = tf.Variable(weights, name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[w_shape[3]]), name='biases')
            conv = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
            self.summary_list.append(tf.summary.histogram('weights', w))
            self.summary_list.append(tf.summary.histogram('biases', b))
            out = tf.add(conv, b, name='output')
            if batch_norm:
                out = tf.layers.batch_normalization(out, training=self.train, name='batch_norm_layer')
            if activation:
                if act_param.get('type') == 'ReLU':
                    act = tf.nn.relu(out, name='ReLu')
                    self.summary_list.append(tf.summary.histogram('ReLU', act))
                    out = act
                elif act_param.get('type') == 'leaky':
                    act = tf.nn.leaky_relu(out, act_param.get('param'), name='Leaky_ReLU')
                    self.summary_list.append(tf.summary.histogram('Leaky_ReLU', act))
                    out = act
            if pooling:
                if pool_param.get('type') == 'max':
                    out = tf.nn.max_pool(out, pool_param.get('kernel'), pool_param.get('strides'),
                                         pool_param.get('padding'),
                                         name='MaxPool')
        return out

    def create_fc_layer(self, x, shape, name, activation=True, dropout=False, act_param=None, dropout_param=None,
                        weight_init='Normal', batch_norm=True):
        """
         This function will create fully connected layer with activation function and dropout under same
         tensorflow name scope.
        :param x: A Tensor. Must be one of the following types: half, bfloat16, float32.
        :param shape: A Tensor. Must have the same type as input.
        :param name: Name for tensorflow name scope.
        :param activation: Boolean value. If true, it will create an activation function in this layer.
        :param dropout: Boolean value. If true, it will create a dropout in this layer.
        :param act_param: dict with keys 'type' and 'param'.
        :param dropout_param: A scalar Tensor with the same type as x. The probability that each element is kept.
        :param weight_init:
        :return:
        """
        if activation and act_param is None:
            act_param = {'type': 'ReLU', 'param': -1}

            # if weight_init != 'Normal':
            #     tf.logging.warning('Currently weight initialization is supported with normal distribution. '
            #                        'Continue with Normal distribution')
            # weight_init = 'Normal'
        with tf.name_scope(name):
            if weight_init == 'Normal':
                weights = tf.truncated_normal(shape, stddev=0.1)
            elif weight_init == 'Xavier':
                weights = tf.contrib.layers.xavier_initializer()
                weights = weights(shape)
                tf.logging.info('Using Xavier init for %s layer' % name)
            else:
                weights = tf.truncated_normal(shape, stddev=0.1)
            w = tf.Variable(weights, name="weights")
            b = tf.Variable(tf.constant(0.1, shape=[shape[1]]), name="biases")
            self.summary_list.append(tf.summary.histogram("weights", w))
            self.summary_list.append(tf.summary.histogram("biases", b))
            out = tf.add(tf.matmul(x, w), b, name='output')
            if batch_norm:
                out = tf.layers.batch_normalization(out, training=self.train, name='Batch_norm_layer')
            if activation:
                if act_param.get('type') == 'ReLU':
                    act = tf.nn.relu(out, name='ReLU')
                    self.summary_list.append(tf.summary.histogram('ReLU', act))
                    out = act
                elif act_param.get('type') == 'leaky':
                    act = tf.nn.leaky_relu(out, act_param.get('param'), name='Leaky ReLU')
                    self.summary_list.append(tf.summary.histogram('Leaky_ReLU', act))
                    out = act
                elif act_param.get('type') == 'sigmoid':
                    act = tf.sigmoid(out, name='Sigmoid')
                    self.summary_list.append(tf.summary.histogram('Sigmoid', act))
                    out = act
            if dropout:
                if dropout_param is not None:
                    out = tf.nn.dropout(out, dropout_param, name='Dropout')
                else:
                    tf.logging.warning('Dropout flag was set to True in layer %s, but parameter was not specified. '
                                       'Continue without dropout layer' % name)
        return out
