# Started at 11:30 19-06-2018

import tensorflow as tf


EMBEDDING_SIZE = 200
VOCAB_SIZE = 10000
SENTENCE_LENGTH_MAX = 100

DEFAULT_TYPE = tf.float32
DEFAULT_INITIALIZER = tf.truncated_normal_initializer(stddev=5e-2)


def project_words(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    assert len(batch_sentences.shape) == 2
    assert batch_sentences.shape[1] == SENTENCE_LENGTH_MAX

    with tf.variable_scope('embedding'):
        word_embeddings = tf.get_variable(name='word_embeddings', dtype=DEFAULT_TYPE, shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                          initializer=DEFAULT_INITIALIZER)
        projected_words = tf.nn.embedding_lookup(params=word_embeddings, ids=batch_sentences)

    assert len(projected_words.shape) == 3
    assert projected_words.shape[0] == batch_sentences.shape[0]
    assert projected_words.shape[1] == SENTENCE_LENGTH_MAX
    assert projected_words.shape[2] == EMBEDDING_SIZE

    return projected_words


def conv(tensor_input, kernel_size, number_filters, stride, name=0):
    """

    :param tensor_input: [batch_size, height, width, channels]
    :param kernel_size:
    :param number_filters:
    :param stride:
    :param name:
    :return:
    """

    assert len(tensor_input.shape) == 4, len(tensor_input.shape)
    with tf.variable_scope('convolution_layer_{}'.format(name)):
        tf_filters = tf.get_variable(name='kernel',
                                     shape=(kernel_size, kernel_size, tensor_input.shape[3], number_filters),
                                     dtype=DEFAULT_TYPE, initializer=DEFAULT_INITIALIZER)
        tf_output = tf.nn.conv2d(tensor_input, tf_filters, strides=[1, stride, stride, 1], padding='VALID')
    assert tf_output.shape[0] == tensor_input.shape[0]
    assert tf_output.shape[3] == number_filters
    return tf_output


def fc(tensor_input, size, name=0):
    assert len(tensor_input.shape) == 2, len(tensor_input.shape)
    with tf.variable_scope('fully_connected_layer_{}'.format(name)):
        tf_weights = tf.get_variable(name='weights',
                                     shape=(tensor_input.shape[1], size),
                                     dtype=DEFAULT_TYPE,
                                     initializer=DEFAULT_INITIALIZER)
        tf_output = tf.matmul(tensor_input, tf_weights)
    assert tf_output.shape[1] == size, tf_output.shape[1]
    return tf_output


def squash_to_output(tensor_input, output_size, name=0):
    with tf.variable_scope('squash_layer_{}'.format(name)):
        tf_output = tf.nn.softmax(tensor_input)
    return tf_output


def inference(batch_sentences):
    """

    :param batch_sentences: [batch_size, sentence_length_max]
    :return:
    """
    projected_input = project_words(batch_sentences)
    projected_input = tf.reshape(tensor=projected_input, shape=list(projected_input.shape) + [1])

    after_conv = conv(projected_input, kernel_size=5, number_filters=100, stride=2, name=0)
    print('After conv 0: ', after_conv.shape)

    after_conv = conv(after_conv, kernel_size=5, number_filters=100, stride=2, name=1)
    print('After conv 1: ', after_conv.shape)

    flatten = tf.reshape(after_conv, [-1, after_conv.shape[1] * after_conv.shape[2] * after_conv.shape[3]])
    print('After flatt: ', flatten.shape)

    after_fc = fc(flatten, 100, name=0)
    print('After fc 0: ', after_fc.shape)

    return after_fc


def loss(batch_sentences, batch_labels):
    """

    :param batch_sentences:
    :param batch_labels: [batch_size]
    :return:
    """
    tf_inference = inference(batch_sentences)
    assert len(tf_inference.shape) == 2, len(tf_inference.shape)
    tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=tf_inference)
    return tf_loss


def optimize(tf_loss):
    pass