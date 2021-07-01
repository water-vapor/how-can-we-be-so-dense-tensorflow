import argparse

from datasets import MNIST
from models import *
from utils import seed_random


def main(dataset_str, model_name_str, batch_size, epochs):
    if dataset_str == 'mnist':
        dataset = MNIST()
    else:
        raise NotImplementedError
    seed_random(0)

    y_train, y_test = dataset.onehot_labels

    if model_name_str == 'sparse_cnn':
        model = build_keras_model(sparse_cnn)
        x_train, x_test = dataset.get_normalized_3d_data()
    elif model_name_str == 'dense_cnn':
        model = build_keras_model(dense_cnn)
        x_train, x_test = dataset.get_normalized_3d_data()
    elif model_name_str == 'hybrid_cnn':
        model = build_keras_model(hybrid_cnn)
        x_train, x_test = dataset.get_normalized_3d_data()
    elif model_name_str == 'sparse_mlp':
        model = build_keras_model(sparse_mlp)
        x_train, x_test = dataset.get_normalized_1d_data()
    elif model_name_str == 'dense_mlp':
        model = build_keras_model(dense_mlp)
        x_train, x_test = dataset.get_normalized_1d_data()
    else:
        raise NotImplementedError

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a single model without noise')
    parser.add_argument('-d', dest='dataset', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('-m', dest='model_name', type=str, default='sparse_cnn', help='Name of the model')
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--no-gpu', dest='no_gpu', action='store_true', default=False, help='Disable GPU')
    args = parser.parse_args()
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    main(args.dataset, args.model_name, args.batch_size, args.epochs)
