import unittest

from preprocessor import Text2Vector
import preprocessor


class Text2VectorTest(unittest.TestCase):

    def test_success(self):
        list_docs = ['Hôm nay, tôi đi học. 12321 ', 'Hôm nay, trời 432 đẹp quá!']
        list_docs = preprocessor.preprocess(list_docs)

        transformer = Text2Vector()
        transformer.fit(list_docs)

        print('Most comment words: ', transformer.get_most_common(10))

        vec = transformer.doc_to_vec(preprocessor.preprocess(['Hôm nay, tôi 332 đi học.', 'Hôm nay, 43 tôi đi chơi.!']))
        print('Vec: ', vec)
        text = transformer.vec_to_doc(vec)
        print('Text: ', text)


if __name__ == '__main__':
    unittest.main()
