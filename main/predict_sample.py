import os
import logging
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd
import model_v1

import utils
from dataset_manager import DatasetManager
import preprocessor


tf.flags.DEFINE_string('ALL_DATASET', '/home/ductri/code/all_dataset/', 'path to all dataset')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size')
tf.flags.DEFINE_string('EXP_NAME', '', 'Experiment name')
tf.flags.DEFINE_string('STEP', '', 'Saved model at step')
tf.flags.DEFINE_string('OUTPUT', 'result.csv', 'CSV output file')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.flags.FLAGS


def predict_sample():
    logging.info('*' * 50)
    logging.info('RUNNING PREDICTION FOR MODEL: %s', FLAGS.EXP_NAME)

    df_test = pd.read_csv(os.path.join(FLAGS.ALL_DATASET, 'entropy_2018', 'test_set.csv'))
    size = FLAGS.BATCH_SIZE
    docs = df_test['sentence'][:size]
    labels = df_test['sentiment'][:size]
    predict(list_sentences=docs, list_labels=labels, output_file=FLAGS.OUTPUT, experiment_name=FLAGS.EXP_NAME, step=FLAGS.STEP)


def predict(list_sentences, output_file, experiment_name, step='', list_labels=[]):
    dataset_manager = DatasetManager()
    dataset_manager.boot()
    list_preprocessed_sentences = preprocessor.preprocess(list_sentences)
    list_vecs = dataset_manager.text2vec.doc_to_vec(list_preprocessed_sentences)
    list_vecs = dataset_manager.equalize_vector_length_to_np(list_vectors=list_vecs, max_length=model_v1.SENTENCE_LENGTH_MAX)
    list_labels = dataset_manager.convert_labels_to_np(list_labels)

    if step == '':
        interesting_checkpoint = tf.train.latest_checkpoint(os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name))
    else:
        interesting_checkpoint = os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name, 'step-{}'.format(step))

    with tf.Graph().as_default() as gr:
        logging.info('-- Restoring graph for model: %s', interesting_checkpoint)
        saver = tf.train.import_meta_graph('{}.meta'.format(interesting_checkpoint))
        logging.info('-- Restored graph for model named: %s', interesting_checkpoint)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as sess:
            saver.restore(sess=sess, save_path=interesting_checkpoint)
            logging.info('-- Restored variables for model named: %s', interesting_checkpoint)

            tf_input = gr.get_tensor_by_name('input/tf_input:0')
            tf_predictions = gr.get_tensor_by_name('prediction:0')

            prediction = sess.run(tf_predictions, feed_dict={
                tf_input: list_vecs
            })

            if len(list_labels) != 0:
                logging.info('-- Report for model: %s', experiment_name)
                logging.info(classification_report(y_true=list_labels, y_pred=prediction))

            result_dict = dict()
            result_dict['sentence'] = list_sentences
            result_dict['pre-processed'] = list_preprocessed_sentences
            result_dict['pre-processed_recover'] = dataset_manager.text2vec.vec_to_doc(list_vecs)
            result_dict['predict'] = prediction

            if len(list_labels) != 0:
                result_dict['label'] = list_labels

            pd.DataFrame(result_dict).to_csv(output_file, index=None)
            logging.debug('Saved result at %s', output_file)


def main(argv=None):
    utils.logging_parameters(FLAGS.EXP_NAME)
    predict_sample()


if __name__ == '__main__':
    utils.setup_logging()
    tf.app.run()
