import tensorflow as tf


class NNTemplate:

    def inference(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError

    def create_conv_layer(self, x, w_shape, name, strides=[1, 1, 1, 1], weight_init='Normal'):
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
            print("Different types of weights initialization is not yet supported")

        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[w_shape[3]]), name='biases')
            conv = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
        return conv + b

    def create_fc_layer(self, x, shape, name):
        """
        This method will create fully connected layer with specified parameters
        :param x: input of the fully connected layer
        :param shape: A 2-D tensor of shape [in_channels, out_channels]
        :param name: Name for name scope.
        :return: output of fully connected layer x*W+b
        """
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(0.1, shape=shape[1]), name="biases")
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
        return tf.matmul(x, w) + b

    def create_activation(self, x, act_type, alpha=0.2):
        """
        This method will create activation function
        :param x: input of the activation function
        :param act_type: type of activation function. Must be 'ReLU' or 'leaky'.
        :param alpha: parameter of the activation function
        :return: activation
        """
        if act_type == 'ReLU':
            act = tf.nn.relu(x)
            tf.summary.histogram('ReLU', act)
            return act
        elif act_type == 'leaky':
            act = tf.nn.leaky_relu(x, alpha)
            tf.summary.histogram('Leaky ReLU', act)
            return act
        else:
            # Change print to logger
            print('Wrong name for activation func was used. ReLU will be used as default')
            act = tf.nn.relu(x)
            tf.summary.histogram('ReLU', act)
            return act

    def create_pooling_layer(self, x, kernel, stride):
        # return tf.nn.max_pool(x, kernel, stride, padding='SAME')
        raise NotImplementedError
