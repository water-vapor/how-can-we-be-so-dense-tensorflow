import tensorflow as tf


def random_binary_mask(tensor_shape, false_percentage, strict=True):
    if not strict:
        binary_mask = tf.random.uniform(tensor_shape) > false_percentage
    else:
        binary_mask = tf.reshape(
            tf.random.shuffle(tf.linspace(0.0, 1.0, tf.reduce_prod(tensor_shape)))
            > false_percentage,
            tensor_shape,
        )
    return binary_mask


def kwinner_mask(inputs, k):
    vals, _ = tf.math.top_k(inputs, k=k)
    kth_elems = tf.reduce_min(vals, axis=-1, keepdims=True)
    boolean_mask = tf.greater_equal(inputs, kth_elems)
    return boolean_mask


def flat_kwinner_mask(inputs, k):
    if len(inputs.shape) > 2:
        input_shape_batchless = inputs.shape[1:].as_list()
        flat_input = tf.reshape(inputs, (-1, tf.reduce_prod(input_shape_batchless)))
        boolean_mask = kwinner_mask(flat_input, k)
        return tf.reshape(boolean_mask, (-1, *input_shape_batchless))
    else:
        return kwinner_mask(inputs, k)


def add_noise(img, eta, strict=True):
    noise_val = tf.reduce_mean(img) + 2 * tf.math.reduce_std(img)
    binary_mask = random_binary_mask(img.shape, eta, strict)
    return tf.where(binary_mask, img, noise_val)
