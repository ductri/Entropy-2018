import pandas as pd
import re
from pyvi import ViTokenizer
from joblib import Parallel, delayed
import collections
import itertools
import numpy as np
import logging
import nltk


nltk.download('punkt')


def replace_url(list_docs):
    def clear_url_text(text):
        URL_PATTERN = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        return re.sub(URL_PATTERN, 'URL', text)

    return [clear_url_text(doc) for doc in list_docs]


def clear_number(list_docs):
    def clear_number_text(text):
        NUMBERS_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?")
        return re.sub(NUMBERS_PATTERN, '', text)

    return [clear_number_text(doc) for doc in list_docs]


class Text2Vector:
    OUT_OF_VOCAB = 'OUT_OF_VOCAB'
    PADDING = 'PADDING'

    def __init__(self):
        self.counts = None
        self.int_to_vocab = None
        self.vocab_to_int = None

    def tokenize(self, text, index=-1):
        """

        :param text:
        :return: list
        """
        if index != -1:
            logging.debug('Tokenize count: %s', index)
        if index == 23730:
            logging.debug('Fucking text: %s', text)
        result = ViTokenizer.tokenize(text).split(' ')

        return result

    def doc_to_vec(self, list_documents):
        logging.debug('-- From doc_to_vec')
        assert isinstance(list_documents, list)
        len_list = len(list_documents)
        tokenized_documents = []
        # for i, doc in enumerate(list_documents):
        #     if i % 100 == 0:
        #         logging.debug('--- Tokenizing: {}\{}, len={}'.format(i, len_list, len(doc)))
        #     tokenized_documents.append(self.__tokenize(doc))
        tokenized_documents = Parallel(n_jobs=1)(delayed(self.tokenize)(doc, index) for index, doc in enumerate(list_documents))
        transformed_documents = Parallel(n_jobs=1)(delayed(self.transform)(doc) for doc in tokenized_documents)
        return transformed_documents

    def vec_to_doc(self, list_vecs):
        assert isinstance(list_vecs, list) or isinstance(list_vecs, np.ndarray)
        return [self.__invert_transform(vec) for vec in list_vecs]

    def fit(self, list_texts):
        logging.debug('-- From fit')
        if self.counts or self.vocab_to_int or self.int_to_vocab:
            raise Exception('"fit" is a one-time function')
        logging.debug('Tokenizing %s documents', len(list_texts))
        list_tokenized_texts = Parallel(n_jobs=1)(delayed(self.tokenize)(doc, index) for index, doc in enumerate(list_texts))
        logging.debug('Tokenize done')
        all_tokens = itertools.chain(*list_tokenized_texts)
        self.counts = collections.Counter(all_tokens)
        logging.debug('Count done')
        self.int_to_vocab = self.__get_vocab()
        self.int_to_vocab = [Text2Vector.PADDING] + self.int_to_vocab + [Text2Vector.OUT_OF_VOCAB]
        self.vocab_to_int = {word: index for index, word in enumerate(self.int_to_vocab)}

    def transform(self, list_tokens):
        if not self.vocab_to_int:
            raise Exception('vocab_to_int is None')

        return [self.vocab_to_int[token] if token in self.vocab_to_int else self.vocab_to_int[Text2Vector.OUT_OF_VOCAB] for token in list_tokens]

    def __invert_transform(self, list_ints):
        """

        :param list_ints:
        :return: A document str
        """
        if not self.int_to_vocab:
            raise Exception('vocab_to_int is None')

        return ' '.join([self.int_to_vocab[int_item] for int_item in list_ints])

    def __get_vocab(self, vocab_size=10000):
        if not self.counts:
            raise Exception('counts is None')
        return [item[0] for item in self.counts.most_common(n=vocab_size)]

    def get_most_common(self, n=10):
        if not self.counts:
            raise Exception('counts is None')
        return self.counts.most_common(n)

    def export_vocab(self, output_file):
        pd.DataFrame({'word': self.int_to_vocab}).to_csv(output_file, index=False, header=False)
        logging.debug('Exported %s words in vocab into file %s', len(self.int_to_vocab), output_file)


def preprocess(list_docs):
    preprocessed_docs = clear_number(list_docs)
    preprocessed_docs = replace_url(preprocessed_docs)
    preprocessed_docs = [doc.lower() for doc in preprocessed_docs]
    preprocessed_docs = [re.sub('[^\w\s]', '', doc) for doc in preprocessed_docs]
    return preprocessed_docs


if __name__ == '__main__':
    docs = ['tuổi trẻ online tuoitre.vn']
    print(replace_url(docs))
