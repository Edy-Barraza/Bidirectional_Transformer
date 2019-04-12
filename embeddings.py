import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    """
    Args:
        vocab_size: int specifying how many words in the vocab
        embedding_size: int specifying the dimension of our token vectors
        initializer_range: float specifying range of our embedding values
        segment_types: int specifying the number of segments that could be fed to model
        max_pos_embeddings: int specifying the maximum sequence length this model will ever consider
        drop_rate: float specifying dropout rate

    """

    def __init__(self, vocab_size, embedding_size, initializer_range=0.02, segment_types=16, max_pos_embeddings=512, drop_rate=.1):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.init_range = initializer_range
        self.segment_types = segment_types
        self.max_pos_embeddings = max_pos_embeddings
        self.drop_rate = drop_rate

    def build(self, input_shape):
        assert input_shape[1] <= self.max_pos_embeddings
        self.batch_size = input_shape[0]
        self.seq_len = input_shape[1]

    def call(self, input_ids, segment_ids=None,training=False):
        """
        Args:
            input_ids: tensor of shape [batch_size,seq_len] with ids of tokens in sequence
            segment_embeddings: tensor of shape [batch_size,seq_len] with ids identifying a pair of concatted text with 0's and 1's
            training: boolean specifying whether we are training or not, meaning whether we apply dropout or not

        Returns:
            tensor of shape [batch_size,seq_len,embedding_size] with embedding
        """

        with tf.variable_scope("token_embeddings"):
            token_embedding_matrix = tf.get_variable(name="token_embedding_matrix",
                                                     shape=[self.vocab_size, self.embedding_size],
                                                     initializer=tf.truncated_normal_initializer(self.init_range))
        token_embeddings = tf.nn.embedding_lookup(token_embedding_matrix, input_ids)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[self.batch_size, self.seq_len], dtype=tf.int32)
        with tf.variable_scope("segment_embeddings"):
            segment_embeddings_matrix = tf.get_variable(name="segment_embedding_matrix",
                                                        shape=[self.segment_types, self.embedding_size],
                                                        initializer=tf.truncated_normal_initializer(
                                                            stddev=self.init_range))
        segment_embeddings = tf.nn.embedding_lookup(segment_embeddings_matrix, segment_ids)

        with tf.variable_scope("position_embeddings"):
            position_embedding_matrix = tf.get_variable(name="position_embedding_matrix",
                                                        shape=[self.max_pos_embeddings, self.embedding_size],
                                                        initializer=tf.truncated_normal_initializer(
                                                            stddev=self.init_range))
        position_embeddings = tf.slice(position_embedding_matrix, [0, 0], [self.seq_len, -1])

        all_embeddings = token_embeddings+segment_embeddings+position_embeddings
        if training:
            return tf.nn.dropout(all_embeddings,rate=self.drop_rate)
        return all_embeddings

