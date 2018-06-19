import tensorflow as tf

import model_v1


BATCH_SIZE = 512


def run():
    with tf.Graph().as_default():
        tf_input = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, model_v1.SENTENCE_LENGTH_MAX])
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])

        tf_inference = model_v1.inference(tf_input)
        tf_loss = model_v1.loss(tf_inference, tf_labels)
        tf_optimizer = model_v1.optimize(tf_loss)

        with tf.Session().as_default() as sess:
            sess.run(tf_optimizer)


if __name__ == '__main__':

    run()
