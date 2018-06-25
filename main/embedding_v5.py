import io
import logging


class Embedding:
    __word_embedding = None

    @staticmethod
    def load_pretrained_vectors(fname):
        if Embedding.__word_embedding is None:
            logging.debug('-- Loading pre-trained word embeeding at %s', fname)
            fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            Embedding.__word_embedding = {}
            for line in fin:
                tokens = line.rstrip().split(' ')
                Embedding.__word_embedding[tokens[0]] = [float(token) for token in tokens[1:]]

        logging.debug('--  Pre-trained word embeeding has been loaded')
        logging.info('-- Pre-trained word vocab size: %s', len(Embedding.__word_embedding))
        return Embedding.__word_embedding
