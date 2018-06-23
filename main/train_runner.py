import tensorflow as tf
import numpy as np
import os
import time
import logging.config
from ruamel.yaml import YAML
from datetime import datetime


import model_v1
from dataset_manager import DatasetManager
import utils


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


tf.flags.DEFINE_string('ALL_DATASET', '/home/ductri/code/all_dataset/', 'path to all dataset')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size')
tf.flags.DEFINE_integer('NUMBER_EPOCHS', 1, 'number of epochs size')
tf.flags.DEFINE_integer('TEST_SIZE', 1000, 'test size')
tf.flags.DEFINE_integer('EMBEDDING_SIZE', 200, 'embedding size')

tf.flags.DEFINE_integer('CONV0_KERNEL_FILTER_SIZE', 5, 'kernel filter size in conv 0')
tf.flags.DEFINE_integer('CONV0_KERNEL_POOLING_SIZE', 2, 'kernel pooling size in conv 0')
tf.flags.DEFINE_integer('CONV0_NUMBER_FILTERS', 10, 'number of filters in conv 0')
tf.flags.DEFINE_float('CONV0_DROPOUT', 0.3, 'dropout rate in conv 0')

tf.flags.DEFINE_integer('CONV1_KERNEL_FILTER_SIZE', 5, 'kernel filter size in conv 1')
tf.flags.DEFINE_integer('CONV1_KERNEL_POOLING_SIZE', 2, 'kernel pooling size in conv 1')
tf.flags.DEFINE_integer('CONV1_NUMBER_FILTERS', 10, 'number of filters in conv 1')
tf.flags.DEFINE_float('CONV1_DROPOUT', 0.3, 'dropout rate in conv 1')

tf.flags.DEFINE_integer('FC0_SIZE', 100, 'output size of fully connected layer 0')

tf.flags.DEFINE_boolean('LOG_DEVICE_PLACEMENT', False, 'display which devices are using')


FLAGS = tf.flags.FLAGS


def run(experiment_name):
    with tf.Graph().as_default() as gr:
        tf_input = tf.placeholder(dtype=tf.int32, shape=[None, model_v1.SENTENCE_LENGTH_MAX])
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[None])

        tf_logits = model_v1.inference(tf_input)
        tf_loss = model_v1.loss(tf_logits, tf_labels)

        tf_optimizer, tf_global_step = model_v1.optimize(tf_loss)
        model_v1.measure_acc(tf_logits, tf_labels)

        tf_all_summary = tf.summary.merge_all()

        tf_train_writer = tf.summary.FileWriter(logdir=os.path.join(CURRENT_DIR, '..', 'summary', 'train_' + experiment_name), graph=gr)
        tf_test_writer = tf.summary.FileWriter(logdir=os.path.join(CURRENT_DIR, '..', 'summary', 'test_' + experiment_name), graph=gr)

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.03)

        logging.info('Graph size: %s', utils.count_trainable_variables())

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                              allow_soft_placement=True,
                                              log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT
                                              )).as_default() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            dataset_manager = DatasetManager()
            dataset_manager.boot()

            for docs, labels in dataset_manager.get_batch(batch_size=FLAGS.BATCH_SIZE, number_epochs=FLAGS.NUMBER_EPOCHS):
                _, global_step = sess.run([tf_optimizer, tf_global_step],
                                          feed_dict={
                                              tf_input: docs,
                                              tf_labels: labels
                                          })
                summary_interval_step = 10
                if global_step % summary_interval_step == 0:
                    logging.debug('Global step: %s', global_step)
                    train_summary_data = sess.run(tf_all_summary, feed_dict={
                                              tf_input: docs,
                                              tf_labels: labels
                                          })
                    tf_train_writer.add_summary(train_summary_data, global_step=global_step)

                if global_step % summary_interval_step == 0:
                    docs_test, labels_test = dataset_manager.get_test_set(FLAGS.TEST_SIZE, is_shuffled=True)
                    test_summary_data = sess.run(tf_all_summary, feed_dict={
                        tf_input: docs_test,
                        tf_labels: labels_test
                    })
                    tf_test_writer.add_summary(test_summary_data, global_step=global_step)

                if global_step % 200 == 0:
                    path_to_save = os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name)
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                    saved_file = saver.save(sess, save_path=path_to_save, global_step=global_step)
                    logging.debug('Saving model at %s', saved_file)


def main(argv=None):
    experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
    logging.info('*' * 50)
    logging.info('--- All parameters for running experiment: %s', experiment_name)
    params = FLAGS.flag_values_dict()
    for key in params:
        logging.info('{}: {}'.format(key, params[key]))
    logging.info('*' * 50)

    run(experiment_name)


def setup_logging():
    yaml = YAML(typ='safe')
    with open(os.path.join(CURRENT_DIR, 'config_logging.yaml'), 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)


if __name__ == '__main__':
    setup_logging()
    tf.app.run()
