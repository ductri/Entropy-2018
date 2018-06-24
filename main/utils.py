import os
import logging.config
from ruamel.yaml import YAML
import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def count_trainable_variables():
    params_count = 0
    for v in tf.trainable_variables():
        v_size = np.prod(v.get_shape().as_list())
        logging.debug('-- -- Variable %s contributes %s parameters', v, v_size)
        params_count += v_size
    return params_count
    # return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def setup_logging():
    yaml = YAML(typ='safe')
    with open(os.path.join(CURRENT_DIR, 'config_logging.yaml'), 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)


def logging_parameters(experiment_name):
    logging.info('*' * 50)
    logging.info('--- All parameters for running experiment: %s', experiment_name)
    params = FLAGS.flag_values_dict()
    for key in params:
        if key != 'h' and key != 'help' and key != 'helpful' and key != 'helpshort' and key!= 'helpfull':
            logging.info('{}: {}'.format(key, params[key]))
    logging.info('*' * 10)
