# Started at 11:30 19-06-2018
import logging
import tensorflow as tf


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
    :return:
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


def __conv(tensor_input, kernel_filter_size, kernel_pooling_size, number_filters, stride, dropout=0.3, name=0):
    """

    :param tensor_input: [batch_size, height, width, channels]
    :param kernel_filter_size:
    :param kernel_pooling_size:
    :param number_filters:
    :param stride:
    :param dropout:
    :param name:
    :return:
    """

    assert len(tensor_input.shape) == 4, len(tensor_input.shape)
    with tf.variable_scope('convolution_layer_{}'.format(name)):
        tf_filters = tf.get_variable(name='kernel',
                                     shape=(kernel_filter_size, kernel_filter_size, tensor_input.shape[3], number_filters),
                                     dtype=DEFAULT_TYPE, initializer=DEFAULT_INITIALIZER())
        tf_output = tf.nn.conv2d(tensor_input, tf_filters, strides=[1, stride, stride, 1], padding='SAME')

        tf_bias = tf.get_variable(name='bias',
                                  shape=[number_filters],
                                  dtype=DEFAULT_TYPE,
                                  initializer=DEFAULT_INITIALIZER())
        tf_output = tf.nn.bias_add(tf_output, tf_bias)

        tf_output = tf.nn.relu(tf_output)

        tf_output = tf.nn.max_pool(tf_output, ksize=[1, kernel_pooling_size, kernel_pooling_size, 1], strides=[1, 1, 1, 1], padding='VALID')

        tf_output = tf.nn.dropout(tf_output, keep_prob=1-dropout)

    assert tf_output.shape[3] == number_filters
    return tf_output


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

    assert tf_output.shape[1] == size, tf_output.shape[1]
    return tf_output


def __squash_to_output(tensor_input, name=0):
    with tf.variable_scope('squash_layer_{}'.format(name)):
        tf_output = tf.nn.softmax(tensor_input)
    return tf_output


def inference(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return: logits
    """
    projected_input = __project_words(batch_sentences)
    projected_input = tf.reshape(tensor=projected_input, shape=[-1] + list(projected_input.shape[1:]) + [1])

    after_conv = __conv(projected_input, kernel_filter_size=5, kernel_pooling_size=2, number_filters=10, stride=2, name=0)
    logging.info('After conv 0: %s', after_conv.shape)

    after_conv = __conv(after_conv, kernel_filter_size=5, kernel_pooling_size=2, number_filters=10, stride=2, name=1)
    logging.info('After conv 1: %s', after_conv.shape)

    flatten = tf.reshape(after_conv, [-1, after_conv.shape[1] * after_conv.shape[2] * after_conv.shape[3]])
    logging.info('After flatt: %s', flatten.shape)

    after_fc = __fc(flatten, 100, name=0)
    logging.info('After fc 0: %s', after_fc.shape)

    after_fc = __fc(after_fc, 3, name=1)
    logging.info('After fc 1: %s', after_fc.shape)

    return after_fc


def loss(tf_logits, batch_labels):
    """

    :param tf_logits: [X, number_classes]
    :param batch_labels: [batch_size]
    :return:
    """
    assert len(tf_logits.shape) == 2, len(tf_logits.shape)
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=tf_logits)
    tf_aggregated_loss = tf.reduce_mean(tf_losses)

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss


def optimize(tf_loss):
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int16, shape=(), initializer=ZERO_INITIALIZER())

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(tf_loss, global_step=tf_global_step)
    return optimizer, tf_global_step


def predict(tf_logits):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    tf_predicts = tf.argmax(tf_logits, axis=1, output_type=tf.int32)
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
