import os
import logging
import tensorflow as tf
from sklearn.metrics import classification_report

import utils
from dataset_manager import DatasetManager


tf.flags.DEFINE_string('ALL_DATASET', '/home/ductri/code/all_dataset/', 'path to all dataset')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size')
tf.flags.DEFINE_string('EXP_NAME', '', 'Experiment name')
tf.flags.DEFINE_string('STEP', '', 'Saved model at step')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.flags.FLAGS


def evaluate(experiment_name, step=''):
    logging.info('*' * 50)
    logging.info('RUNNING EVALUATION FOR MODEL: %s', experiment_name)
    if step == '':
        interesting_checkpoint = tf.train.latest_checkpoint(os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name))
    else:
        interesting_checkpoint = os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name, 'step-{}'.format(step))
    dataset_manager = DatasetManager()
    dataset_manager.boot()

    with tf.Graph().as_default() as gr:
        logging.info('-- Restoring graph for model: %s', interesting_checkpoint)
        saver = tf.train.import_meta_graph('{}.meta'.format(interesting_checkpoint))
        logging.info('-- Restored graph for model named: %s', interesting_checkpoint)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as sess:
            saver.restore(sess=sess, save_path=interesting_checkpoint)
            logging.info('-- Restored variables for model named: %s', interesting_checkpoint)
            list_predictions = []
            list_labels = []
            for docs, labels in dataset_manager.get_test_by_batch(batch_size=FLAGS.BATCH_SIZE):
                tf_input = gr.get_tensor_by_name('input/tf_input:0')
                tf_predictions = gr.get_tensor_by_name('prediction:0')

                prediction = sess.run(tf_predictions, feed_dict={
                    tf_input: docs
                })
                list_predictions.extend(prediction)
                list_labels.extend(labels)
                logging.debug('-- Prediction length: %s/%s', len(list_predictions), dataset_manager.test_y.shape[0])
            logging.info('-- Report for model: %s', experiment_name)
            logging.info(classification_report(y_true=list_labels, y_pred=list_predictions))


def main(argv=None):
    experiment_name = FLAGS.EXP_NAME
    utils.logging_parameters(experiment_name)
    evaluate(experiment_name, step=FLAGS.STEP)


if __name__ == '__main__':
    utils.setup_logging()
    tf.app.run()
