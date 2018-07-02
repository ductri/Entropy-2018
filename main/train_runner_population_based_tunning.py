import tensorflow as tf
import os
import logging.config
from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import json

from dataset_manager import DatasetManager
import utils
from model_population_based_tunning import Model
import model_population_based_tunning


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('MODEL_VERSION', 'v1', 'version of model')

tf.flags.DEFINE_string('ALL_DATASET', '/home/ductri/code/all_dataset/', 'path to all dataset')
tf.flags.DEFINE_string('WORD_EMBEDDING_FILE', '/home/ductri/code/all_dataset/', 'path to all dataset')
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
tf.flags.DEFINE_float('FC0_DROPOUT', 0.3, 'dropout rate in FC 0')
tf.flags.DEFINE_float('FC1_DROPOUT', 0.3, 'dropout rate in FC 1')

tf.flags.DEFINE_integer('NUM_HIDDEN', 100, 'size of LSTM cell')

tf.flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate')
tf.flags.DEFINE_boolean('IS_INCLUDED_REGULARIZATION', False, 'is_included_regularization')


tf.flags.DEFINE_boolean('LOG_DEVICE_PLACEMENT', False, 'display which devices are using')
tf.flags.DEFINE_float('GPU', 0.5, 'size of LSTM cell')


def run(experiment_name):
    BEST_THRES = 3
    WORST_THRES = 3
    POPULATION_STEPS = 500
    ITERATIONS = 100
    POPULATION_SIZE = 10
    accuracy_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
    l1_scale_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
    best_accuracy_hist = np.zeros((POPULATION_STEPS,))
    best_l1_scale_hist = np.zeros((POPULATION_STEPS,))

    with tf.Graph().as_default() as gr:

        with tf.variable_scope('input'):
            tf_input = tf.placeholder(dtype=tf.int32, shape=[None, model_population_based_tunning.SENTENCE_LENGTH_MAX], name='tf_input')
            tf_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_labels')

        models = [create_model(i, is_included_regularization=FLAGS.IS_INCLUDED_REGULARIZATION) for i in range(10)]
        # It will help us with creation of different scope_name for each model
        for index, model in enumerate(models):
            with tf.variable_scope(str(index)):
                model.boot(tf_input, tf_labels)

        logging.info('Graph size: %s', utils.count_trainable_variables())

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.GPU)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                              allow_soft_placement=True,
                                              log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT
                                              )).as_default() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            dataset_manager = DatasetManager()
            dataset_manager.boot()

            dataset_generator = dataset_manager.get_batch(batch_size=FLAGS.BATCH_SIZE, number_epochs=10 * FLAGS.NUMBER_EPOCHS)
            for i in range(POPULATION_STEPS):

                # Copy best
                sess.run([m.get_copy_from_op(models[0]) for m in models[-WORST_THRES:]])
                # Perturb others
                sess.run([m.l1_scale_perturb_op for m in models[BEST_THRES:]])
                # Training
                for _ in range(ITERATIONS):
                    docs, labels = next(dataset_generator)
                    sess.run([m.tf_optimizer for m in models], feed_dict={
                                              tf_input: docs,
                                              tf_labels: labels
                                          })
                docs, labels = next(dataset_generator)
                # Evaluate
                l1_scales = sess.run({m: m.l1_scale for m in models})
                accuracies = sess.run({m: m.tf_acc for m in models}, feed_dict={
                                              tf_input: docs,
                                              tf_labels: labels
                                          })
                models.sort(key=lambda m: accuracies[m], reverse=True)
                # Logging
                best_accuracy_hist[i] = accuracies[models[0]]
                best_l1_scale_hist[i] = l1_scales[models[0]]
                for m in models:
                    l1_scale_hist[m.model_id, i] = l1_scales[m]
                    accuracy_hist[m.model_id, i] = accuracies[m]
            with open('temp', 'w') as output_f:
                json.dump({'accuracy_hist': accuracy_hist,
                           'l1_scale_hist': l1_scale_hist,
                           'best_accuracy_hist': best_accuracy_hist,
                           'best_l1_scale_hist': best_l1_scale_hist
                           }, output_f)


def create_model(model_id, **kwargs):
    with tf.variable_scope(str(model_id)):
        return Model(model_id, **kwargs)


def main(argv=None):
    experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
    logging.info('Tunning with name: %s', experiment_name)
    utils.logging_parameters(experiment_name)
    run(experiment_name)


if __name__ == '__main__':
    utils.setup_logging()
    tf.app.run()
