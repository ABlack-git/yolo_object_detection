import tensorflow as tf


def f1():
    a = tf.constant([[[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    return a


def f2():
    b = tf.constant([[[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    return b


def test_1():
    max_a = tf.reduce_max(f1(), axis=2)
    max_b = tf.reduce_max(f2(), axis=2)
    is_obj = tf.cond(tf.equal(tf.reduce_min(tf.to_float(tf.equal(max_a, max_b))), 1),
                     f1,
                     f2)

    # a = tf.constant([[1, 2, 3], [4, 5, 6], [9, 8, 7], [23, 45, 67]], dtype=tf.float32)
    size = tf.shape(f1())
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    s, m_a, m_b, o = sess.run([size, max_a, max_b, is_obj])
    print('Tensor size is:\n' + str(s))
    print('Max_a\n' + str(m_a))
    print('Max_b\n' + str(m_b))
    print('Output\n' + str(o))
    sess.close()


def test_2():
    b = f2()
    t = tf.to_float(tf.not_equal(b, 1))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    t_o = sess.run(t)
    print(str(t_o))
    sess.close()


if __name__ == '__main__':
    test_2()
