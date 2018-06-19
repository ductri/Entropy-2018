import tensorflow as tf
import numpy as np


def count_trainable_variables():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
