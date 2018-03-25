import tensorflow as tf


class NNTemplate:

    def _inference(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def optimize(self, epochs):
        raise NotImplementedError

    def save(self, path, name):
        raise NotImplementedError

    def restore(self, path, meta):
        raise NotImplementedError

    def _create_conv_layer(self, x, w_shape, name, strides=[1, 1, 1, 1], weight_init='Normal'):
        """
        This method will create a convolutional layer with specified parameters.
        :param x: input of the convolutional layer.
        :param w_shape: A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels].
        :param name: Name for name scope.
        :param strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input.
        :param weight_init: Strategy of weights initialization. Can be 'Normal' or 'Xavier'.
        :return: A Tensor that represents input after covolution plus biases.
        """
        if weight_init != 'Normal':
            # Change print to logger
            tf.logging.warning("Different types of weights initialization is not yet supported")

        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[w_shape[3]]), name='biases')
            conv = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            out = tf.add(conv, b, name='output')
        return out

    def _create_fc_layer(self, x, shape, name):
        """
        This method will create fully connected layer with specified parameters
        :param x: input of the fully connected layer
        :param shape: A 2-D tensor of shape [in_channels, out_channels]
        :param name: Name for name scope.
        :return: output of fully connected layer x*W+b
        """
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(0.1, shape=[shape[1]]), name="biases")
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            out = tf.add(tf.matmul(x, w), b, name='output')
        return out

    def _create_activation(self, x,act_type, alpha=0.2):
        """
        This method will create activation function
        :param x: input of the activation function
        :param act_type: type of activation function. Must be 'ReLU' or 'leaky'.
        :param alpha: parameter of the activation function
        :return: activation
        """
        if act_type == 'ReLU':
            act = tf.nn.relu(x, name='ReLU')
            tf.summary.histogram('ReLU', act)
            return act
        elif act_type == 'leaky':
            act = tf.nn.leaky_relu(x, alpha, name='Leaky_ReLU')
            tf.summary.histogram('Leaky ReLU', act)
            return act
        else:
            tf.logging.warning('Wrong name for activation func was used. ReLU will be used as default')
            act = tf.nn.relu(x, name='ReLU')
            tf.summary.histogram('ReLU', act)
            return act

    def _create_pooling_layer(self, x, kernel, stride):
        """
        Create max pool layer
        :param x: input value
        :param kernel:  A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
        :param stride: A 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension
        of the input tensor.
        :return: A Tensor of format specified by data_format. The max pooled output tensor.
        """
        return tf.nn.max_pool(x, kernel, stride, padding='SAME')
