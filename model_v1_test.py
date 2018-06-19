import tensorflow as tf
import unittest
import numpy as np

import model_v1
import utils


class ModelV1Test(unittest.TestCase):

    def test_project_words(self):
        batch_size = 512
        with tf.Graph().as_default():
            tf_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, model_v1.SENTENCE_LENGTH_MAX])
            tf_output = model_v1.project_words(tf_input)
            with tf.Session() as sess:
                initializer = tf.global_variables_initializer()
                sess.run(initializer)

                output = sess.run(tf_output, feed_dict={tf_input: np.ones((batch_size, model_v1.SENTENCE_LENGTH_MAX))})
                print(np.mean(output))

    def test_conv(self):
        batch_size = 512
        with tf.Graph().as_default():
            tf_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10, 11, 1])
            tf_output = model_v1.conv(tf_input, kernel_size=5, number_filters=100, stride=2, name=0)
            with tf.Session() as sess:
                initializer = tf.global_variables_initializer()
                sess.run(initializer)

                output = sess.run(tf_output, feed_dict={tf_input: np.ones((batch_size, 10, 11, 1))})
                print(np.mean(output))

    def test_inference(self):
        batch_size = 512
        with tf.Graph().as_default():
            tf_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, model_v1.SENTENCE_LENGTH_MAX])
            tf_output = model_v1.inference(tf_input)
            print('Graph size: {}'.format(utils.count_trainable_variables()))
            with tf.Session() as sess:
                initializer = tf.global_variables_initializer()
                sess.run(initializer)

                output = sess.run(tf_output, feed_dict={tf_input: np.ones((batch_size, model_v1.SENTENCE_LENGTH_MAX))})
                print(np.mean(output))


if __name__ == '__main__':
    unittest.main()