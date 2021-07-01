import numpy as np
import tensorflow as tf


def seed_random(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
