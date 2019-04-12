import tensorflow as tf
import attention
import pw_ffn


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Transformer from BERT paper https://arxiv.org/abs/1810.04805

    Args:
        input_shape: list [batch_sz, seq_len, hidden_sz]
        num_heads: number of heads in multi-headed attention
        mask: list or numpy array of shape (batch_sz,seq_len) with 1's for kept words and 0's for masked positions
        drop_rate: dropout rate
        inner_ffn_dim: int specifying projection dimension of first dense layer of the ffn
        output_ffn_dim: int specifying output dimension of entire ffn; must be original size of input to ffn

    """

    def __init__(self, input_shape, inner_ffn_dim, output_ffn_dim, num_heads=8, mask=None, drop_rate=.1):
        super(TransformerEncoderBlock, self).__init__()

        self.attention_layer = attention.Multi_Head_Attention(input_shape, num_heads, mask, drop_rate)
        self.ffn = pw_ffn.PWFFN(inner_ffn_dim, output_ffn_dim)
        self.first_layernorm = lambda x: tf.contrib.layers.layer_norm(x, begin_norm_axis=-1, begin_params_axis=-1)
        self.second_layernorm = lambda y: tf.contrib.layers.layer_norm(y, begin_norm_axis=-1, begin_params_axis=-1)

    def assign_mask(self,mask):
        self.attention_layer.assign_mask(mask)

    def call(self, x, training=False):
        x_attended = self.attention_layer(x, None, training)
        x = self.first_layernorm(x + x_attended)
        x_ffn = self.ffn(x)
        return self.second_layernorm(x + x_ffn)


