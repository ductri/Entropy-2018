import os
import pandas as pd
import logging
import numpy as np
import json
import logging.config
from ruamel.yaml import YAML
import tensorflow as tf

# import preprocessor
import preprocessor_padding as preprocessor
import model_v1

FLAGS = tf.flags.FLAGS

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class DatasetManager:

    ENTROPY_DATASET = 'entropy_2018'
    BINARY_TRAINING_FILE = 'binary_training_data'
    BINARY_TEST_FILE = 'binary_test_data'

    LABEL_MAPPING = {
        'positive': 0,
        'neutral': 1,
        'negative': 2
    }

    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.text2vec = preprocessor.Text2Vector()

    def boot(self):
        self.__maybe_boot_train()
        self.__maybe_boot_test()
        logging.info('Booting done!')
        logging.info('Train shape: %s', self.train_X.shape)
        logging.info('Test shape: %s', self.test_X.shape)

    def get_batch(self, batch_size, number_epochs):
        logging.info('There will be approximately {} steps'.format(self.train_X.shape[0] // batch_size * number_epochs))
        for _ in range(number_epochs):
            shuffle_index = list(range(self.train_X.shape[0]))
            np.random.shuffle(shuffle_index)
            temp_X = self.train_X[shuffle_index]
            temp_y = self.train_y[shuffle_index]
            for j in range((self.train_X.shape[0] // batch_size) - 1):
                yield temp_X[j*batch_size:(j+1)*batch_size, :], temp_y[j*batch_size:(j+1)*batch_size]

    def get_test_set(self, size, is_shuffled=True):
        if is_shuffled:
            shuffled_index = list(range(size))
            np.random.shuffle(shuffled_index)
            return self.test_X[shuffled_index], self.test_y[shuffled_index]

        return self.test_X[:size], self.test_y[:size]

    def get_test_by_batch(self, batch_size):
        logging.info('There will be approximately {} steps'.format(self.test_X.shape[0] // batch_size))
        for i in range((self.test_X.shape[0] // batch_size) - 1):
            yield self.test_X[i*batch_size:(i+1)*batch_size, :], self.test_y[i*batch_size:(i+1)*batch_size]

    def get_vocab_size(self):
        return len(self.text2vec.int_to_vocab)

    def convert_labels_to_np(self, list_text_labels):
        return np.array([DatasetManager.LABEL_MAPPING[label] for label in list_text_labels])

    def equalize_vector_length_to_np(self, list_vectors, max_length):
        assert isinstance(list_vectors, list)
        assert isinstance(list_vectors[0], list)
        for i in range(len(list_vectors)):
            if len(list_vectors[i]) < max_length:
                list_vectors[i] += [self.text2vec.vocab_to_int[preprocessor.Text2Vector.PADDING]] * (max_length - len(list_vectors[i]))
            elif len(list_vectors[i]) > max_length:
                list_vectors[i] = self.__simple_cut(list_vectors[i], max_length)
        return np.array(list_vectors)

    def __simple_cut(self, list_tokens, max_length):
        return list_tokens[:max_length]

    def __maybe_boot_train(self):
        path_to_file = os.path.join(CURRENT_DIR, 'data', DatasetManager.BINARY_TRAINING_FILE)
        if os.path.isfile(path_to_file):
            logging.info('Binary train exists')
            with open(path_to_file, 'r') as input_file:
                data = json.load(input_file)
                self.train_X = np.array(data['X'])
                self.train_y = np.array(data['y'])
                self.text2vec = preprocessor.Text2Vector()
                self.text2vec.int_to_vocab = data['int_to_vocab']
                self.text2vec.vocab_to_int = data['vocab_to_int']

        else:
            logging.info('Loading data ...')
            df_train = pd.read_csv(os.path.join(FLAGS.ALL_DATASET, DatasetManager.ENTROPY_DATASET, 'training_set.csv'))
            logging.info('Pre-processing training data ...')
            train_list_docs = preprocessor.preprocess(list(df_train['sentence']))

            logging.info('Vectoring training text ...')
            self.text2vec.fit(train_list_docs)
            train_list_vectors = self.text2vec.doc_to_vec(train_list_docs)

            self.train_X = self.equalize_vector_length_to_np(list_vectors=train_list_vectors, max_length=model_v1.SENTENCE_LENGTH_MAX)

            self.train_y = self.convert_labels_to_np(df_train['sentiment'])

            assert self.train_X.shape[0] == self.train_y.shape[0]

            with open(os.path.join(CURRENT_DIR, 'data', DatasetManager.BINARY_TRAINING_FILE), 'w') as output_file:
                json.dump({'X': self.train_X.tolist(),
                           'y': self.train_y.tolist(),
                           'int_to_vocab': self.text2vec.int_to_vocab,
                           'vocab_to_int': self.text2vec.vocab_to_int
                           }, output_file)

    def __maybe_boot_test(self):
        path_to_file = os.path.join(CURRENT_DIR, 'data',  DatasetManager.BINARY_TEST_FILE)
        if os.path.isfile(path_to_file):
            logging.info('Binary test exists')
            with open(path_to_file, 'r') as input_file:
                data = json.load(input_file)
                self.test_X = np.array(data['X'])
                self.test_y = np.array(data['y'])
        else:
            logging.info('Loading data ...')
            df_test = pd.read_csv(os.path.join(FLAGS.ALL_DATASET, DatasetManager.ENTROPY_DATASET, 'test_set.csv'))
            logging.info('Pre-processing test data ...')
            test_list_docs = preprocessor.preprocess(list(df_test['sentence']))

            logging.info('Vectoring test text ...')

            test_list_vectors = self.text2vec.doc_to_vec(test_list_docs)

            self.test_X = self.equalize_vector_length_to_np(list_vectors=test_list_vectors, max_length=model_v1.SENTENCE_LENGTH_MAX)

            self.test_y = self.convert_labels_to_np(df_test['sentiment'])

            assert self.test_X.shape[0] == self.test_y.shape[0]

            with open(os.path.join(CURRENT_DIR, 'data', DatasetManager.BINARY_TEST_FILE), 'w') as output_file:
                json.dump({'X': self.test_X.tolist(), 'y': self.test_y.tolist()}, output_file)


if __name__ == '__main__':
    yaml = YAML(typ='safe')
    with open(os.path.join(CURRENT_DIR, 'config_logging.yaml'), 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)

    dataset_manager = DatasetManager()
    dataset_manager.boot()

    print('Test')

    for i in range(10):
        index = np.random.randint(dataset_manager.train_X.shape[0])
        print('*' * 30)
        print('Sentence: ', dataset_manager.text2vec.vec_to_doc([dataset_manager.train_X[index]]))
        print('Sentiment: ', dataset_manager.train_y[index])

    print('*' * 100)
    print('*' * 100)
    for docs, sentiments in dataset_manager.get_batch(batch_size=10, number_epochs=1):
        print(dataset_manager.text2vec.vec_to_doc(docs))
        break

    print('*' * 100)
    print('*' * 100)
    print('Vocab size: ', dataset_manager.get_vocab_size())
