import tensorflow as tf


class MNIST:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = tf.constant(x_train, dtype=tf.float32)
        self.x_test = tf.constant(x_test, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.float32)
        self.y_test = tf.constant(y_test, dtype=tf.float32)

    def _normalize_preprocess(self, preprocessing=None):
        x_train, x_test = self.x_train / 255., self.x_test / 255.
        if preprocessing:
            x_train = tf.vectorized_map(preprocessing, x_train)
            x_test = tf.vectorized_map(preprocessing, x_test)
        return x_train, x_test

    def get_normalized_1d_data(self, preprocessing=None):
        x_train, x_test = self._normalize_preprocess(preprocessing=preprocessing)
        return tf.reshape(x_train, (-1, 28 * 28)), tf.reshape(x_test, (-1, 28 * 28))

    def get_normalized_2d_data(self, preprocessing=None):
        x_train, x_test = self._normalize_preprocess(preprocessing=preprocessing)
        return x_train, x_test

    def get_normalized_3d_data(self, preprocessing=None):
        x_train, x_test = self._normalize_preprocess(preprocessing=preprocessing)
        return tf.expand_dims(x_train, -1), tf.expand_dims(x_test, -1)

    @property
    def onehot_labels(self):
        return tf.keras.utils.to_categorical(self.y_train, 10), tf.keras.utils.to_categorical(self.y_test, 10)
