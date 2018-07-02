# Started at 11:30 19-06-2018
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib as tfc
import logging
import numpy as np


VOCAB_SIZE = 10002
SENTENCE_LENGTH_MAX = 150

DEFAULT_TYPE = tf.float32

FLAGS = tf.flags.FLAGS


def DEFAULT_INITIALIZER():
    return tf.truncated_normal_initializer(stddev=5e-2)


def ZERO_INITIALIZER():
    return tf.zeros_initializer()


def inference(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    word_embeddings = __project_words(batch_sentences)
    word_embeddings = tf.unstack(word_embeddings, SENTENCE_LENGTH_MAX, axis=1) # SENTENCE_LENGTH_MAX tensors which shape=[batch_size, embedding_size]

    with tf.variable_scope('LSTM'):
        lstm_cell = rnn.BasicLSTMCell(FLAGS.NUM_HIDDEN, forget_bias=1.0)
        outputs, states = rnn.static_rnn(cell=lstm_cell, inputs=word_embeddings, dtype=tf.float32)
    l1_reg = __get_regularizer()
    tf_logits = tf.layers.dense(outputs[-1], units=FLAGS.FC0_SIZE, activation=tf.nn.relu, kernel_regularizer=l1_reg)
    tf_logits = tf.layers.dense(tf_logits, units=3, kernel_regularizer=l1_reg)

    return tf_logits


def loss(tf_logits, batch_labels):
    """

    :param tf_logits: [X, number_classes]
    :param batch_labels: [batch_size]
    :return:
    """
    assert len(tf_logits.shape) == 2, len(tf_logits.shape)
    assert tf_logits.shape[1] == 3
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=tf_logits)
    tf_aggregated_loss = tf.reduce_mean(tf_losses) + tf.losses.get_regularization_loss()

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss


def optimize(tf_loss):
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=ZERO_INITIALIZER())
    apply_gradient_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.LEARNING_RATE).minimize(tf_loss, global_step=tf_global_step)

    return apply_gradient_op, tf_global_step


def predict(tf_logits):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    tf_predicts = tf.argmax(tf_logits, axis=1, output_type=tf.int32, name='prediction')
    return tf_predicts


def measure_acc(tf_logits, batch_labels):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :param batch_labels: [batch_size]
    :return:
    """
    tf_predicts = predict(tf_logits)
    tf_acc = tf.reduce_mean(tf.cast(tf.equal(tf_predicts, batch_labels), 'float'))
    tf.summary.scalar(name='accuracy', tensor=tf_acc)
    return tf_acc


def __get_regularizer():
    return tfc.layers.l1_regularizer(2 ** FLAGS.L1_SCALE)


def __project_words(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return: [batch_size, sentence_length_max, embedding_size]
    """
    assert len(batch_sentences.shape) == 2
    assert batch_sentences.shape[1] == SENTENCE_LENGTH_MAX

    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        word_embeddings = tf.get_variable(name='word_embeddings', dtype=DEFAULT_TYPE,
                                          shape=[VOCAB_SIZE, FLAGS.EMBEDDING_SIZE],
                                          initializer=DEFAULT_INITIALIZER())
        projected_words = tf.nn.embedding_lookup(params=word_embeddings, ids=batch_sentences)

    assert len(projected_words.shape) == 3

    assert projected_words.shape[1] == SENTENCE_LENGTH_MAX
    assert projected_words.shape[2] == FLAGS.EMBEDDING_SIZE

    return projected_words
