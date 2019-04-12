import tensorflow as tf
import transformer_block
import embeddings


class Transformer(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 vocab_size,
                 num_blocks=12,
                 inner_ffn_dim=3072,
                 output_ffn_dim=768,
                 embedding_size=768,
                 initializer_range=0.02,
                 segment_types=16,
                 max_pos_embeddings=512,
                 num_heads=12,
                 drop_rate=.1,
                 mask=None):

        """
        Args:
            input_shape: int list [batch_sz, seq_len, embedding_size]
            vocab_size: int specifying how many words in the vocab
            num_blocks= int specifying number of encoder transformer blocks
            inner_ffn_dim: int specifying projection dimension of first dense layer of the ffn
            output_ffn_dim: int specifying output dimension of entire ffn; must be original size of input to ffn
            embedding_size: int specifying the dimension of our token vectors
            initializer_range: float specifying range of our embedding values
            segment_types: int specifying the number of segments that could be fed to model
            max_pos_embeddings: int specifying the maximum sequence length this model will ever consider
            num_heads: int number of heads in multi-headed attention
            drop_rate: float dropout rate
            mask: list or numpy array of shape (batch_sz,seq_len) with 1's for kept words and 0's for masked positions

        """

        super(Transformer, self).__init__()

        self.model_embeddings = embeddings.Embedding(vocab_size=vocab_size,
                                                     embedding_size=embedding_size,
                                                     initializer_range=initializer_range,
                                                     segment_types=segment_types,
                                                     max_pos_embeddings=max_pos_embeddings,
                                                     drop_rate=drop_rate)

        transformer_blocks = []
        for _ in range(num_blocks):
            transformer_blocks.append(
                transformer_block.TransformerEncoderBlock(input_shape=input_shape,
                                                          inner_ffn_dim=inner_ffn_dim,
                                                          output_ffn_dim=output_ffn_dim,
                                                          num_heads=num_heads,
                                                          mask=mask,
                                                          drop_rate=drop_rate))

        self.transformer_blocks = transformer_blocks

    def assign_mask(self, mask):
        for block in self.transformer_blocks:
            block.assign_mask(mask)

    def call(self, input_ids, segment_ids=None, training=False):
        embedded_input = self.model_embeddings(input_ids, segment_ids, training)
        output = embedded_input
        for block in self.transformer_blocks:
            output = block(output, training)

        return output



