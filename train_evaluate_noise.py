import argparse

from datasets import MNIST
from models import *
from ops import add_noise
from utils import seed_random


def main(dataset_str, model_name_str, batch_size, epochs):
    if dataset_str == 'mnist':
        dataset = MNIST()
    else:
        raise NotImplementedError
    seed_random(0)

    y_train, y_test = dataset.onehot_labels

    if model_name_str == 'sparse_cnn':
        sparse_model = build_keras_model(sparse_cnn)
    elif model_name_str == 'hybrid_cnn':
        sparse_model = build_keras_model(hybrid_cnn)
    else:
        raise NotImplementedError
    dense_model = build_keras_model(dense_cnn)
    x_train, x_test = dataset.get_normalized_3d_data()
    x_test_noise = {0: x_test}
    for noise_level in range(5, 60, 5):
        x_test_noise[noise_level] = dataset.get_normalized_3d_data(
            preprocessing=lambda x: add_noise(x, noise_level / 100, strict=False))[1]
    print('Training dense_cnn')
    dense_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)
    print(f'Training {model_name_str}')
    sparse_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)
    dense_acc = [dense_model.evaluate(x_test_noise[i], y_test)[1] for i in range(0, 60, 5)]
    sparse_acc = [sparse_model.evaluate(x_test_noise[i], y_test)[1] for i in range(0, 60, 5)]
    for noise_level, acc in zip(range(0, 60, 5), dense_acc):
        print(f'dense_cnn on noise level {noise_level}%, test accuracy: {acc}')
    for noise_level, acc in zip(range(0, 60, 5), sparse_acc):
        print(f'{model_name_str} on noise level {noise_level}%, test accuracy: {acc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare dense cnn and hybrid/sparse cnn\'s performance with noise')
    parser.add_argument('-d', dest='dataset', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('-m', dest='sparse_model_name', type=str,
                        default='hybrid_cnn', help='Name of the model to compare with (hybrid_cnn or sparse_cnn)')
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--no-gpu', dest='no_gpu', action='store_true', default=False, help='Disable GPU')
    args = parser.parse_args()
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    main(args.dataset, args.sparse_model_name, args.batch_size, args.epochs)
