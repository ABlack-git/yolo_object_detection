import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from network.MNIST.nn_template import NNTemplate


class MnistNN(NNTemplate):

    def __init__(self, restore=False):
        #
        self.sess = tf.Session()

        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        # placeholders

        # Critical variables and operations
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        # initialize model
        if not restore:
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="input")
            self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            self.labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
            self.dropout = tf.placeholder(tf.float32, name="dropout_prob")

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self._inference()
            self._loss()
            self._optimizer()
            self._accuracy()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        else:
            self.x = None
            self.labels = None
            self.dropout = None
            self.global_step = None
            self.saver = None

    def _inference(self):
        """
        This function initializes model and set self.predictions to be the output of the neural network
        :return:
        """
        if not self.predictions:
            conv1 = super()._create_conv_layer(self.x_image, [5, 5, 1, 32], 'Conv_layer_1')
            act1 = super()._create_activation(conv1, 'ReLU')
            pool1 = super()._create_pooling_layer(act1, [1, 2, 2, 1], [1, 2, 2, 1])
            conv2 = super()._create_conv_layer(pool1, [5, 5, 32, 64], 'Conv_layer_2')
            pool2 = super()._create_pooling_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1])
            act2 = super()._create_activation(pool2, 'ReLU')
            flatened = tf.reshape(act2, [-1, 7 * 7 * 64])
            fc1 = super()._create_fc_layer(flatened, [7 * 7 * 64, 1024], 'FC_layer_1')
            act3 = super()._create_activation(fc1, 'ReLU')
            dropout = tf.nn.dropout(act3, self.dropout, name='Dropout')
            self.predictions = super()._create_fc_layer(dropout, [1024, 10], 'FC_layer_2')

    def _loss(self):
        if not self.loss:
            with tf.name_scope("loss"):
                # I want to define placeholder here and add after reduce_mean.
                # Maybe try place it before reduce_mean
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predictions), name='Loss')
                tf.summary.scalar("cross_entropy", self.loss)

    def _accuracy(self):
        if not self.accuracy:
            with tf.name_scope("Accuracy"):
                prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='Accuracy')
                tf.summary.scalar("Accuracy", self.accuracy)

    def _optimizer(self):
        if not self.optimizer:
            with tf.name_scope("train"):
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss, self.global_step, name='Optimizer')

    def optimize(self, summary_dir, epochs=1000):

        writer = tf.summary.FileWriter(summary_dir)
        summaries = tf.summary.merge_all()
        writer.add_graph(self.sess.graph, tf.train.global_step(self.sess, self.global_step))
        tf.logging.info("Starting training. Current global step is: %s",
                        tf.train.global_step(self.sess, self.global_step))
        setup = self.sess.partial_run_setup([self.predictions, self.optimizer, self.loss],
                                            [self.x, self.labels, self.dropout])
        for i in range(epochs):
            batch = self.mnist.train.next_batch(50)
            if i % 100 == 0:
                s = self.sess.run(summaries,
                                  feed_dict={self.x: batch[0], self.labels: batch[1], self.dropout: float(0.5)})
                writer.add_summary(s, tf.train.global_step(self.sess, self.global_step))
                train_accuracy = self.sess.run(self.accuracy, feed_dict={self.x: batch[0], self.labels: batch[1],
                                                                         self.dropout: 0.5})
                tf.logging.info(
                    "Accuracy at step %s is %f" % (tf.train.global_step(self.sess, self.global_step), train_accuracy))

            self.sess.partial_run(setup, self.predictions,
                                  feed_dict={self.x: batch[0], self.labels: batch[1], self.dropout: 0.5})
            self.sess.partial_run(setup, self.loss)
            self.sess.partial_run(setup, self.optimizer)
            # self.sess.run(self.optimizer, feed_dict={self.x: batch[0], self.labels: batch[1], self.dropout: 0.5})
        self.save('weights', 'mnist_model')
        writer.close()
        tf.logging.info("Training has finished")

    def test_model(self):
        accuracy = self.sess.run(self.accuracy, feed_dict={
            self.x: self.mnist.test.images, self.labels: self.mnist.test.labels,
            self.dropout: 1.0})
        tf.logging.info("Accuracy of the model is %g" % accuracy)

    def close_all(self):
        self.sess.close()

    def open_sess(self):
        if not self.sess:
            self.sess = tf.Session()

    def set_logger_verbosity(self, verbosity=tf.logging.INFO):
        tf.logging.set_verbosity(verbosity)

    def save(self, path, name):
        """
        Saves parameters of the network
        :param path: directory where to store parameters
        :param name: name of the model
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if any(file.endswith('.meta') for file in os.listdir(path)):
            self.saver.save(self.sess, os.path.join(path, name),
                            global_step=tf.train.global_step(self.sess, self.global_step), write_meta_graph=False)
        else:
            self.saver.save(self.sess, os.path.join(path, name),
                            global_step=tf.train.global_step(self.sess, self.global_step))

    def restore(self, path, meta):
        self.saver = tf.train.import_meta_graph(meta)
        try:
            self.saver.restore(self.sess, save_path=path)
            self.x = tf.get_default_graph().get_tensor_by_name('input:0')
            self.labels = tf.get_default_graph().get_tensor_by_name('labels:0')
            self.dropout = tf.get_default_graph().get_tensor_by_name('dropout_prob:0')
            self.predictions = tf.get_default_graph().get_tensor_by_name('FC_layer_2/output:0')
            self.loss = tf.get_default_graph().get_tensor_by_name('loss/Loss:0')
            self.accuracy = tf.get_default_graph().get_tensor_by_name('Accuracy:0')
            self.optimizer = tf.get_default_graph().get_operation_by_name('train/Optimizer')
            self.global_step = tf.get_default_graph().get_tensor_by_name('global_step:0')
        except KeyError as e:
            tf.logging.fatal("Restoring was not successful.")
            tf.logging.fatal(e)
            exit(1)

        tf.logging.info("Model restored. Global step is %s" % tf.train.global_step(self.sess, self.global_step))


def main():
    model = MnistNN()
    model.set_logger_verbosity()
    model.optimize('mnist_summaries/2', 500)
    # model.optimize('mnist_summaries/2', 500)
    model.test_model()
    model.close_all()


def test_restore():
    model = MnistNN(True)
    model.set_logger_verbosity()
    model.restore(
        'weights/mnist_model-1000',
        'weights/mnist_model-500.meta')
    model.test_model()
    model.optimize('mnist_summaries/1', 500)
    model.test_model()
    model.close_all()


if __name__ == '__main__':
    main()
