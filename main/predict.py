import os
import logging
import tensorflow as tf
import pandas as pd

import utils
from dataset_manager import DatasetManager
import preprocessor
import model_v6

tf.flags.DEFINE_string('ALL_DATASET', '', 'path to all dataset, not required')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size, not require')
tf.flags.DEFINE_string('EXP_NAME', '2018-06-26T16:25:00', 'Experiment name')
tf.flags.DEFINE_string('STEP', '73600', 'Saved model at step')

tf.flags.DEFINE_string('INPUT_FILE', 'input.csv', 'CSV file should not contain header')
tf.flags.DEFINE_string('OUTPUT_FILE', 'output.csv', 'Output file')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.flags.FLAGS


def predict(list_docs, experiment_name, step='', batch_size=64):

    logging.info('*' * 50)
    logging.info('RUNNING PREDICT FOR MODEL: %s', experiment_name)
    if step == '':
        interesting_checkpoint = tf.train.latest_checkpoint(os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name))
    else:
        interesting_checkpoint = os.path.join(CURRENT_DIR, '..', 'checkpoint', experiment_name, 'step-{}'.format(step))
    dataset_manager = DatasetManager()
    dataset_manager.boot()

    list_preprocessed_sentences = preprocessor.preprocess(list_docs)
    list_vecs = dataset_manager.text2vec.doc_to_vec(list_preprocessed_sentences)
    list_vecs = dataset_manager.equalize_vector_length_to_np(list_vectors=list_vecs,
                                                             max_length=model_v6.SENTENCE_LENGTH_MAX)

    with tf.Graph().as_default() as gr:
        logging.info('-- Restoring graph for model: %s', interesting_checkpoint)
        saver = tf.train.import_meta_graph('{}.meta'.format(interesting_checkpoint))
        logging.info('-- Restored graph for model named: %s', interesting_checkpoint)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as sess:
            saver.restore(sess=sess, save_path=interesting_checkpoint)
            logging.info('-- Restored variables for model named: %s', interesting_checkpoint)
            list_predictions = []

            num_steps = len(list_vecs) // batch_size
            logging.info('There will be %s steps', num_steps + 1)
            for i in range(num_steps + 1):
                tf_input = gr.get_tensor_by_name('input/tf_input:0')
                tf_predictions = gr.get_tensor_by_name('prediction:0')

                prediction = sess.run(tf_predictions, feed_dict={
                    tf_input: list_vecs[i*batch_size: (i+1)*batch_size]
                })
                list_predictions.extend([dataset_manager.LABEL_UNMAPPING[p] for p in prediction])

            return list_predictions


def main(argv=None):
    experiment_name = FLAGS.EXP_NAME
    utils.logging_parameters(experiment_name)
    df_input = pd.read_csv(os.path.join(CURRENT_DIR, '..', 'input', FLAGS.INPUT_FILE), sep='\t', header=None)
    logging.info('I am going to classify %s posts. Please confirm this figure', df_input.shape[0])
    list_predictions = predict(list_docs=list(df_input[0]), experiment_name=FLAGS.EXP_NAME, step=FLAGS.STEP, batch_size=FLAGS.BATCH_SIZE)
    pd.DataFrame({'predict': list_predictions, 'sen': list(df_input[0])}).to_csv(os.path.join(CURRENT_DIR, '..', 'output', FLAGS.OUTPUT_FILE), header=None, index=None)
    logging.info('Result have been saved at %s', FLAGS.OUTPUT_FILE)


if __name__ == '__main__':
    utils.setup_logging()
    tf.app.run()
