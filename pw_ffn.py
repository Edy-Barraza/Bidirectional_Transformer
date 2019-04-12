import tensorflow as tf


class PWFFN(tf.keras.layers.Layer):
    """
    Point-Wise Feed Forward Network From Attention is All You Need https://arxiv.org/abs/1706.03762

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        inner_ffn_dim: int specifying projection dimension of first dense layer of the ffn
        output_ffn_dim: int specifying output dimension of entire ffn; must be original size of input to ffn
    """

    def __init__(self, inner_ffn_dim, output_ffn_dim):
        super(PWFFN, self).__init__()
        self.inner_proj = tf.keras.layers.Dense(inner_ffn_dim)
        self.output_proj = tf.keras.layers.Dense(output_ffn_dim)

    def call(self, x):
        intermed = self.inner_proj(x)
        return self.output_proj(intermed)