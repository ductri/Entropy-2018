import tensorflow as tf
import numpy as np
import os
import time

from main import model_v1

BATCH_SIZE = 512


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def run(name):
    with tf.Graph().as_default() as gr:
        tf_input = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, model_v1.SENTENCE_LENGTH_MAX])
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])

        tf_inference = model_v1.inference(tf_input)
        tf_loss = model_v1.loss(tf_inference, tf_labels)

        tf_optimizer, tf_global_step = model_v1.optimize(tf_loss)

        tf_all_summary = tf.summary.merge_all()

        tf_writer = tf.summary.FileWriter(logdir=os.path.join(CURRENT_DIR, 'summary', str(name), str(int(time.time()))), graph=gr)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for _ in range(1000):
                _, global_step = sess.run([tf_optimizer, tf_global_step],
                                          feed_dict={
                                              tf_input: np.random.randint(low=0, high=model_v1.VOCAB_SIZE, size=(BATCH_SIZE, model_v1.SENTENCE_LENGTH_MAX)),
                                              tf_labels: np.random.randint(low=0, high=3, size=BATCH_SIZE)
                                          })
                print('global step: ', global_step)
                if global_step % 10 == 0:
                    summary_data = sess.run(tf_all_summary, feed_dict={
                                              tf_input: np.random.randint(low=0, high=model_v1.VOCAB_SIZE, size=(BATCH_SIZE, model_v1.SENTENCE_LENGTH_MAX)),
                                              tf_labels: np.random.randint(low=0, high=3, size=BATCH_SIZE)
                                          })
                    tf_writer.add_summary(summary_data, global_step=global_step)


if __name__ == '__main__':
    run('test-1')
