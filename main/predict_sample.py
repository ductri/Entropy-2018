import os
import logging
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd

import utils
from dataset_manager import DatasetManager


tf.flags.DEFINE_string('ALL_DATASET', '/home/ductri/code/all_dataset/', 'path to all dataset')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size')
tf.flags.DEFINE_string('EXP_NAME', '', 'Experiment name')
tf.flags.DEFINE_string('STEP', '', 'Saved model at step')
tf.flags.DEFINE_string('OUTPUT', 'result.csv', 'CSV output file')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.flags.FLAGS


def predict_sample(experiment_name, step=''):
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

            docs, labels = dataset_manager.get_test_set(FLAGS.BATCH_SIZE)
            tf_input = gr.get_tensor_by_name('input/tf_input:0')
            tf_predictions = gr.get_tensor_by_name('prediction:0')

            prediction = sess.run(tf_predictions, feed_dict={
                tf_input: docs
            })

            logging.info('-- Report for model: %s', experiment_name)
            logging.info(classification_report(y_true=labels, y_pred=prediction))
            sentences = dataset_manager.text2vec.vec_to_doc(docs)
            pd.DataFrame({
                'sentence': sentences,
                'label': labels,
                'predict': prediction
            }).to_csv(FLAGS.OUTPUT, index=None)
            logging.debug('Saved')


def main(argv=None):
    experiment_name = FLAGS.EXP_NAME
    utils.logging_parameters(experiment_name)
    predict_sample(experiment_name, step=FLAGS.STEP)


if __name__ == '__main__':
    utils.setup_logging()
    tf.app.run()
