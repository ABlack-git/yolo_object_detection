import tensorflow as tf


class NNTemplate:

    def inference(self, x):
        raise NotImplementedError

    def loss(self, predictions, labels):
        raise NotImplementedError

    def optimize(self, epochs):
        raise NotImplementedError

    def save(self, sess, path, name):
        raise NotImplementedError

    def restore(self, sess, path, meta):
        raise NotImplementedError

    def create_conv_layer(self, x, w_shape, name, strides=None, activation=True, pooling=True, act_param=None,
                          pool_param=None, weight_init='Normal'
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

        if weight_init != 'Normal':
            tf.logging.warning('Currently weight initialization is supported with normal distribution. '
                               'Continue with Normal distribution')

        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[w_shape[3]]), name='biases')
            conv = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            out = tf.add(conv, b, name='output')
            if activation:
                if act_param.get('type') == 'ReLU':
                    act = tf.nn.relu(out, name='ReLu')
                    tf.summary.histogram('ReLU', act)
                    out = act
                elif act_param.get('type') == 'leaky':
                    act = tf.nn.leaky_relu(out, act_param.get('param'), name='Leaky_ReLU')
                    tf.summary.histogram('Leaky_ReLU', act)
                    out = act
            if pooling:
                if pool_param.get('type') == 'max':
                    tf.nn.max_pool(out, pool_param.get('kernel'), pool_param.get('strides'), pool_param.get('padding'))
        return out

    def create_fc_layer(self, x, shape, name, activation=True, dropout=False, act_params=None, dropout_param=None,
                        weight_init='Normal'):
        """
         This function will create fully connected layer with activation function and dropout under same
         tensorflow name scope.
        :param x: A Tensor. Must be one of the following types: half, bfloat16, float32.
        :param shape: A Tensor. Must have the same type as input.
        :param name: Name for tensorflow name scope.
        :param activation: Boolean value. If true, it will create an activation function in this layer.
        :param dropout: Boolean value. If true, it will create a dropout in this layer.
        :param act_params: dict with keys 'type' and 'param'.
        :param dropout_param: A scalar Tensor with the same type as x. The probability that each element is kept.
        :param weight_init:
        :return:
        """
        if activation and act_params is None:
            act_params = {'type': 'ReLU', 'param': -1}

        if weight_init != 'Normal':
            tf.logging.warning('Currently weight initialization is supported with normal distribution. '
                               'Continue with Normal distribution')
            weight_init = 'Normal'
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(0.1, shape=[shape[1]]), name="biases")
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            out = tf.add(tf.matmul(x, w), b, name='output')
            if activation:
                if act_params.get('type') == 'ReLU':
                    act = tf.nn.relu(out, name='ReLU')
                    tf.summary.histogram('ReLU', act)
                    out = act
                elif act_params.get('type') == 'leaky':
                    act = tf.nn.leaky_relu(out, act_params.get('param'))
                    tf.summary.histogram('Leaky_ReLU', act)
                    out = act
            if dropout:
                if dropout_param is not None:
                    out = tf.nn.dropout(out, dropout_param, name='Dropout')
                else:
                    tf.logging.warning('Dropout flag was set to True in layer %s, but parameter was not specified. '
                                       'Continue without dropout layer' % name)
        return out
