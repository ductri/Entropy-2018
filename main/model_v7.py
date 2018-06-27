# Started at 11:30 19-06-2018
import logging
import tensorflow as tf
from tensorflow.contrib import rnn


VOCAB_SIZE = 10002
SENTENCE_LENGTH_MAX = 150

DEFAULT_TYPE = tf.float32

FLAGS = tf.flags.FLAGS


def DEFAULT_INITIALIZER():
    return tf.truncated_normal_initializer(stddev=5e-2)


def ZERO_INITIALIZER():
    return tf.zeros_initializer()


def __project_words(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return: [batch_size, sentence_length_max, embedding_size]
    """
    assert len(batch_sentences.shape) == 2
    assert batch_sentences.shape[1] == SENTENCE_LENGTH_MAX

    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        word_embeddings = tf.get_variable(name='word_embeddings', dtype=DEFAULT_TYPE, shape=[VOCAB_SIZE, FLAGS.EMBEDDING_SIZE],
                                          initializer=DEFAULT_INITIALIZER())
        projected_words = tf.nn.embedding_lookup(params=word_embeddings, ids=batch_sentences)

    assert len(projected_words.shape) == 3

    assert projected_words.shape[1] == SENTENCE_LENGTH_MAX
    assert projected_words.shape[2] == FLAGS.EMBEDDING_SIZE

    return projected_words

def __fc(tensor_input, size, name=0):
    assert len(tensor_input.shape) == 2, len(tensor_input.shape)
    with tf.variable_scope('fully_connected_layer_{}'.format(name)):
        tf_weights = tf.get_variable(name='weights',
                                     shape=(tensor_input.shape[1], size),
                                     dtype=DEFAULT_TYPE,
                                     initializer=DEFAULT_INITIALIZER())
        tf_output = tf.matmul(tensor_input, tf_weights)

        tf_bias = tf.get_variable(name='bias', shape=[size], dtype=DEFAULT_TYPE, initializer=ZERO_INITIALIZER())
        tf_output = tf.nn.bias_add(tf_output, tf_bias)
        reg = tf.nn.l2_loss(tf_weights)
    assert tf_output.shape[1] == size, tf_output.shape[1]
    return tf_output, reg


def loss(tf_logits, batch_labels, tf_reg):
    """

    :param tf_logits: [X, number_classes]
    :param batch_labels: [batch_size]
    :return:
    """
    assert len(tf_logits.shape) == 2, len(tf_logits.shape)
    assert tf_logits.shape[1] == 3
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=tf_logits)
    tf_aggregated_loss = tf.reduce_mean(tf_losses)
    tf_losses_with_reg = tf_aggregated_loss + FLAGS.REG_RATE * tf_reg

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    tf.summary.scalar(name='loss_reg', tensor=tf_losses_with_reg)
    return tf_losses_with_reg


def optimize(tf_loss):
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=ZERO_INITIALIZER())

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.LEARNING_RATE).minimize(tf_loss, global_step=tf_global_step)
    return optimizer, tf_global_step


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


def inference(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    word_embeddings = __project_words(batch_sentences)
    word_embeddings = tf.unstack(word_embeddings, SENTENCE_LENGTH_MAX, axis=1) # SENTENCE_LENGTH_MAX tensors which shape=[batch_size, embedding_size]

    with tf.variable_scope('LSTM'):
        fw_lstm_cell = rnn.BasicLSTMCell(FLAGS.NUM_HIDDEN, forget_bias=1.0)
        bw_lstm_cell = rnn.BasicLSTMCell(FLAGS.NUM_HIDDEN, forget_bias=1.0)

        outputs, _, _ = rnn.static_bidirectional_rnn(fw_lstm_cell, bw_lstm_cell, word_embeddings, dtype=tf.float32)
    tf_logits, reg1 = __fc(tensor_input=outputs[-1], size=FLAGS.FC0_SIZE, name=0)
    tf_logits, reg2 = __fc(tensor_input=tf_logits, size=3, name=1)
    reg = reg1 + reg2
    return tf_logits, reg

