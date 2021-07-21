import tensorflow as tf
import numpy as np

tfl = tf.keras.layers
tfi = tf.keras.initializers


class DeltaOrthogonal(tfi.Initializer):
    def __init__(self, gain=1.0, seed=None):
        self.orthogonal_init = tfi.Orthogonal(gain=gain, seed=seed)

    def __call__(self, shape, dtype=None):
        assert shape[0] == shape[1]
        weight = np.zeros(shape)
        mid = shape[0] // 2
        weight[mid, mid, :, :] = self.orthogonal_init(shape=shape[2:], dtype=dtype)
        return tf.cast(weight, dtype=dtype)

def run_layers(layers, inputs, **kwargs):
    out = inputs
    for layer in layers:
        out = layer(out, **kwargs)
    return out


class LayersWrapper(tfl.Layer):
    def __init__(self, layers):
        super(LayersWrapper, self).__init__()
        self._wrapper_layers = layers

    def call(self, inputs):
        return run_layers(self._wrapper_layers, inputs)


class BilinearLayer(tfl.Layer):
    def __init__(self, initializer=None):
        super(BilinearLayer, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
        assert len(input_shape) == 2, 'input must be a list of two tensors'
        feature_dim1 = input_shape[0][-1]
        feature_dim2 = input_shape[1][-1]
        self.kernel = self.add_weight("kernel", shape=[feature_dim1, feature_dim2],
                                      initializer=self.initializer, trainable=True)
        super(BilinearLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(tf.matmul(inputs[0], self.kernel), inputs[1], transpose_b=True)

class Gather(tfl.Layer):
    def __init__(self, indices):
        super(Gather, self).__init__()
        self._indices = indices

    def call(self, inputs):
        return tf.gather(inputs, self._indices, axis=-1)


class DenseRecursiveSum(tfl.Layer):
    def __init__(self, max_plan_len, out_dims, kernel_initializer=None, bias_initializer=None,):
        super(DenseRecursiveSum, self).__init__()
        self._max_plan_len = max_plan_len
        self._out_dims = out_dims
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._accumulator = tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)),
                                                num_lower=0, num_upper=-1)

    def build(self, input_shape):
        self._in_filters = input_shape[-1]
        self._kernel = self.add_weight(shape=[self._max_plan_len, self._in_filters, self._out_dims],
                                       initializer=self._kernel_initializer,
                                       trainable=True)
        self._bias = self.add_weight(shape=[1, self._max_plan_len, self._out_dims],
                                     initializer=self._bias_initializer or tfi.Zeros(),
                                     trainable=True)
        super(DenseRecursiveSum, self).build(input_shape)

    def call(self, inputs):
        representations = tf.einsum('ijk,jkl->ijl', inputs, self._kernel) + self._bias
        accumulated_representations = tf.einsum('ijl,jk->ikl', representations, self._accumulator)
        return accumulated_representations
