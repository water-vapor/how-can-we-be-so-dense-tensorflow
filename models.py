import tensorflow as tf

from layers import SparseDense, KWinner, SparseConv2D


def dense_mlp():
    inputs = tf.keras.Input(shape=(784,))
    dense1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    logits = tf.keras.layers.Dense(10)(dense2)
    return inputs, logits


def sparse_mlp():
    inputs = tf.keras.Input(shape=(784,))
    dense1 = SparseDense(128, activation='linear')(inputs)
    act1 = KWinner(k=40)(dense1)
    dense2 = SparseDense(64, activation='linear')(act1)
    act2 = KWinner(k=20)(dense2)
    logits = tf.keras.layers.Dense(10)(act2)
    return inputs, logits


def dense_cnn():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv2d = tf.keras.layers.Conv2D(30, (3, 3))(inputs)
    pooled = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv2d)
    act1 = tf.keras.layers.ReLU()(pooled)
    flattened = tf.keras.layers.Flatten()(act1)
    dense2 = tf.keras.layers.Dense(150, activation='relu')(flattened)
    logits = tf.keras.layers.Dense(10)(dense2)
    return inputs, logits


def sparse_cnn():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv2d = SparseConv2D(30, (3, 3))(inputs)
    pooled = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv2d)
    act1 = KWinner(k=400)(pooled)
    flattened = tf.keras.layers.Flatten()(act1)
    dense2 = SparseDense(150)(flattened)
    act2 = KWinner(k=50)(dense2)
    logits = tf.keras.layers.Dense(10)(act2)
    return inputs, logits


def hybrid_cnn():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv2d = tf.keras.layers.Conv2D(30, (3, 3))(inputs)
    pooled = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv2d)
    act1 = KWinner(k=400)(pooled)
    flattened = tf.keras.layers.Flatten()(act1)
    dense2 = SparseDense(150)(flattened)
    act2 = tf.keras.layers.ReLU()(dense2)
    logits = tf.keras.layers.Dense(10)(act2)
    return inputs, logits


def build_keras_model(model_name):
    inputs, logits = model_name()
    model = tf.keras.models.Model(inputs, logits)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model
