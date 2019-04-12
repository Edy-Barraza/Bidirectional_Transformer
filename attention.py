from __future__ import division

import tensorflow as tf




class Multi_Head_Attention(tf.keras.layers.Layer):  # tf.layers.Layer
    """
    Multi-Headed Attention Layer from Attention is All You Need https://arxiv.org/abs/1706.03762

    MutliHead(Q,K,V)=concat(head_1,...,head_h)
    head_i = Attention(Q W_i^Q,K W_i^K,V W_i^V)
    Attention(Q,K,V) = softmax(Q K.T/root(d_k))V

    Args:
        input_shape: list [batch_sz, seq_len, hidden_sz]
        num_heads: number of heads in multi-headed attention
        mask: list or numpy array of shape (batch_sz,seq_len) with 1's for kept words and 0's for masked positions
        drop_rate: dropout rate

    """

    def __init__(self, input_shape=[None, None, None], num_heads=8, mask=None,
                 drop_rate=.1):  # (self, batch_sz, seq_len, hidden_sz, num_heads=8,drop_rate=.1):
        assert len(input_shape) == 3

        super(Multi_Head_Attention, self).__init__()
        self.batch_sz = int(input_shape[0])
        self.seq_len = int(input_shape[1])
        self.hidden_sz = int(input_shape[2])
        self.num_heads = num_heads
        self.sub_dim = self.hidden_sz // self.num_heads
        self.drop_rate = drop_rate

        if mask is None:
            self.mask = mask
        if mask is not None:
            mask = tf.constant(mask, dtype=tf.float32)
            mask = tf.reshape(mask, [self.batch_sz, 1, self.seq_len])
            broadcast_ones = tf.ones(shape=[self.batch_sz, self.seq_len, 1], dtype=tf.float32)
            mask = broadcast_ones * mask
            self.mask = tf.expand_dims(mask, axis=[1])

        self.q_projector = tf.keras.layers.Dense(self.hidden_sz, activation=tf.nn.relu, name="q_projector")
        self.k_projector = tf.keras.layers.Dense(self.hidden_sz, activation=tf.nn.relu, name="k_projector")
        self.v_projector = tf.keras.layers.Dense(self.hidden_sz, activation=tf.nn.relu, name="v_projector")

        self.out_projector = tf.layers.Dense(self.hidden_sz, activation="relu", name="out_projector")

        assert self.hidden_sz % self.num_heads == 0

    def assign_mask(self, mask):
        """
        Args:
            mask: list or numpy array of shape (batch_sz,seq_len)
        creates mask to block out words we wish to guess during training
        mask ends with shape [batch_sz,1,seq_len,seq_len]
        """
        if mask is None:
            self.mask = None
        else:
            mask = tf.constant(mask, dtype=tf.float32)
            mask = tf.reshape(mask, [self.batch_sz, 1, self.seq_len])

            broadcast_ones = tf.ones(shape=[self.batch_sz, self.seq_len, 1], dtype=tf.float32)

            mask = broadcast_ones * mask
            self.mask = tf.expand_dims(mask, axis=[1])

    def split_heads(self, x):
        """
        Args:
            x: tensor of shape [batch_sz, seq_len, hidden_sz/num_heads]

        Returns:
            tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]


        We project to lower dimensional vector space by projecting to same dimensional vector space with dense
        layer and then splitting output tensor along hidden_sz
        """
        x = tf.reshape(x, shape=[self.batch_sz, self.seq_len, self.num_heads, self.sub_dim])
        return tf.transpose(x, [0, 2, 1, 3])  # [batch_sz, num_heads, seq_len, sub_dim]

    def join_heads(self, x):
        """
        Args:
            x: tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]

        Returns:
            tensor of shape [batch_sz, seq_len, hidden_sz]

        """
        x = tf.transpose(x, [0, 2, 1, 3])  # [batch_sz, seq_length, num_heads, sub_dim]
        return tf.reshape(x, [self.batch_sz, self.seq_len, self.hidden_sz])

    def dot_product_attention(self, Q, K, V, training):
        """
        Args:
            Q: tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]
            K: tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]
            V: tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]

        Returns:
            tensor of shape [batch_sz, num_heads, seq_len, hidden_sz/num_heads]

        """

        logits = tf.matmul(Q, K, transpose_b=True)
        scaled_logits = logits / (self.sub_dim ** -.5)

        # masks are 1.0 for words we want to attend to and zero for words we wish it ignore
        # we add zero to words we wish to attend to and subtract 1,000 for words we wish to ignore
        # so that the softmax provides no weight to those positions, so that we ignore those words
        if self.mask is not None:
            adder = (1.0 - self.mask) * -10000.0
            scaled_logits += adder

        weights = tf.nn.softmax(scaled_logits)
        # OG transformer paper suggests dropping out words entirely
        if training:
            weights = tf.nn.dropout(weights, rate=self.drop_rate)
        return tf.matmul(weights, V)

    def call(self, x, y=None, training=False):
        """
        Args:
            x: tensor of shape [batch_sz, seq_len, hidden_sz]
            y: None or tensor of shape [batch_sz, seq_len, hidden_sz]

        Returns:
            tensor of shape [batch_sz, seq_len, hidden_sz]


        """
        if y is None:
            y = x

        Q = self.q_projector(x)
        K = self.k_projector(y)
        V = self.v_projector(y)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_vectors = self.dot_product_attention(Q, K, V, training)
        attention_vectors = self.join_heads(attention_vectors)

        output = self.out_projector(attention_vectors)
        return output





