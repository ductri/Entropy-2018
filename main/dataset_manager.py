import os
import pandas as pd
import logging
import numpy as np
import json
import logging.config
from ruamel.yaml import YAML

from main import preprocessor

ALL_DATASET = '/home/ductri/code/all_dataset/'
ENTROPY_DATASET = os.path.join(ALL_DATASET, 'entropy_2018')
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class DatasetManager:
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
        logging.info('Loading test data ...')
        df_test = pd.read_csv(os.path.join(ENTROPY_DATASET, 'test_data.csv'))
        logging.info('Test size: %s', df_test.shape[0])

        logging.info('Pre-processing test data ...')
        test_list_docs = preprocessor.preprocess(list(df_test['sentence']))

        logging.info('Vectoring test text ...')
        test_list_vectors = self.text2vec.doc_to_vec(test_list_docs)

        self.test_X = self.__equalize_vector_length(test_list_vectors)

        self.test_y = self.__convert_labels(df_test['sentiment'])

        with open(os.path.join(CURRENT_DIR, 'data', DatasetManager.BINARY_TEST_FILE), 'bw') as output_file:
            json.dump({'X': self.test_X, 'y': self.test_y}, output_file)

        logging.info('Booting done!')


    def get_batch(self, batch_size, number_epochs):
        for _ in number_epochs:
            shuffe_index = range(self.train_X.shape[0])
            np.random.shuffle(shuffe_index)
            temp_X = self.train_X[shuffe_index]
            temp_y = self.train_y[shuffe_index]
            for i in range(self.train_X.shape[0]//batch_size-1):
                yield temp_X[i:(i+1)*batch_size, :], temp_y[i:(i+1)*batch_size]

    def get_test_set(self):
        return self.test_X, self.test_y

    def __convert_labels(self, list_text_labels):
        return [DatasetManager.LABEL_MAPPING[label] for label in list_text_labels]

    def __equalize_vector_length(self, list_vectors, max_length):
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

    def __maybe_boot(self, type):
        if type == 'train':
            file_name = DatasetManager.BINARY_TRAINING_FILE
        else:
            file_name = DatasetManager.BINARY_TEST_FILE

        path_to_file = os.path.join(CURRENT_DIR, 'data', file_name)
        if os.path.isfile(path_to_file):
            with open(path_to_file, 'rb') as input_file:
                data = json.load(input_file)
                return data['X'], data['y']
        else:
            logging.info('Loading data ...')
            df_train = pd.read_csv(os.path.join(ENTROPY_DATASET, 'training_data.csv'))
            logging.info('Training size: %s', df_train.shape[0])

            logging.info('Pre-processing training data ...')
            train_list_docs = preprocessor.preprocess(list(df_train['sentence']))

            logging.info('Vectoring training text ...')
            self.text2vec.fit(train_list_docs)
            train_list_vectors = self.text2vec.doc_to_vec(train_list_docs)

            self.train_X = self.__equalize_vector_length(train_list_vectors)

            self.train_y = self.__convert_labels(df_train['sentiment'])

            with open(os.path.join(CURRENT_DIR, 'data', DatasetManager.BINARY_TRAINING_FILE), 'bw') as output_file:
                json.dump({'X': self.train_X, 'y': self.train_y}, output_file)


if __name__ == '__main__':
    yaml = YAML(typ='safe')
    with open('config_logging.yaml', 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)

    dataset_manager = DatasetManager()
    dataset_manager.boot()

