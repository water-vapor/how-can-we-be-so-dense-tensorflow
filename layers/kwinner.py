import tensorflow as tf

from ops import flat_kwinner_mask


class KWinner(tf.keras.layers.Layer):
    def __init__(
            self, k, boost=True, alpha=1 / 1000, beta=1.0, k_inference_factor=1.0, **kwargs
    ):
        """

        Args:
            k: active neurons to keep
            boost: whether to use boosted version of k-winner
            alpha: 1/T, where T=1000, ref: https://doi.org/10.3389/fncom.2017.00111
            beta: boost factor
            k_inference_factor: multiply k by this factor in inference
            **kwargs:
        """
        super().__init__(**kwargs)
        self.k = k
        self.boost = boost
        self.alpha = alpha
        self.beta = beta
        self.k_inference_factor = k_inference_factor
        self.filters = None
        self.units = None
        self.target_duty_cycle = None

    def build(self, input_shape):
        super().build(input_shape)
        if not self.units:
            self.filters = input_shape[-1]
            self.units = tf.reduce_prod(input_shape[1:])
        self.target_duty_cycle = tf.cast(self.k / self.units, dtype=self.dtype)
        self.duty_cycle = self.add_weight(shape=(self.filters,), initializer=tf.keras.initializers.Zeros(),
                                          trainable=False)

    def call(self, inputs, training=False, **kwargs):
        if not training:
            k = tf.cast(self.k * self.k_inference_factor, dtype=tf.int32)
        else:
            k = self.k

        if self.units <= k:
            return tf.nn.relu(inputs)
        if self.boost:
            boost_term = tf.exp(self.beta * (self.target_duty_cycle - self.duty_cycle))
            boosted_inputs = inputs * boost_term
            boolean_mask = flat_kwinner_mask(boosted_inputs, k)
            if training:
                # formula according to the paper
                current_duty = tf.reduce_mean(tf.cast(boolean_mask, dtype=self.dtype),
                                              axis=tf.range(0, boolean_mask.shape.ndims - 1))
                self.duty_cycle.assign((1 - self.alpha) * self.duty_cycle + self.alpha * current_duty)
                # alternative formula from author's code, assumes dim=2
                # however, using batch_size in calculation causes multiple problems in keras,
                # and has no obvious benefits on results
                # current_duty = tf.reduce_sum(tf.cast(boolean_mask, dtype=self.dtype),
                # axis=tf.range(0,boolean_mask.shape.ndims-1))
                # self.duty_cycle.assign((1 - inputs.shape[0] * self.alpha) * self.duty_cycle
                # + self.alpha * current_duty)

        else:
            boolean_mask = flat_kwinner_mask(inputs, k)

        return tf.where(boolean_mask, inputs, tf.zeros_like(inputs))
