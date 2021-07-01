import tensorflow as tf

from ops import random_binary_mask


class SparseDense(tf.keras.layers.Dense):
    def __init__(self, units, sparsity=0.5, strict_sparsity=True, **kwargs):
        super().__init__(units, **kwargs)
        self.sparsity = sparsity
        self.strict_sparsity = strict_sparsity
        self.binary_mask = None

    def build(self, input_shape):
        super().build(input_shape)
        self.binary_mask = random_binary_mask(
            self.kernel.shape, self.sparsity, self.strict_sparsity
        )

    def call(self, inputs):
        self.kernel.assign(
            tf.where(self.binary_mask, self.kernel, tf.zeros_like(self.kernel))
        )
        return super().call(inputs)


class SparseConv2D(tf.keras.layers.Conv2D):
    def __init__(
            self, filters, kernel_size, sparsity=0.5, strict_sparsity=True, **kwargs
    ):
        super().__init__(filters, kernel_size, **kwargs)
        self.sparsity = sparsity
        self.strict_sparsity = strict_sparsity
        self.binary_mask = None

    def build(self, input_shape):
        super().build(input_shape)
        self.binary_mask = random_binary_mask(
            self.kernel.shape, self.sparsity, self.strict_sparsity
        )

    def call(self, inputs):
        self.kernel.assign(
            tf.where(self.binary_mask, self.kernel, tf.zeros_like(self.kernel))
        )
        return super().call(inputs)
