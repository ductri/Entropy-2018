import tensorflow as tf

class Model:
    def __init__(self):
        self.scope_name = tf.get_default_graph().get_name_scope()

def create_model():
    with tf.variable_scope(None, 'model'):
        return Model()


if __name__ == '__main__':
    models = [create_model() for _ in range(10)]

