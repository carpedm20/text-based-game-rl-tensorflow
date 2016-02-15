import tensorflow as tf

from .model import Model

class LSTMDQN(Model):
  def __init__(self):
    pass

  def build_model(self):
    inputs = tf.placeholder(tf.float32, [None, self.seq_length])
    word_indices = tf.split(1, self.seq_length, tf.expand_dims(self.inputs, -1))

    embed = tf.get_variable(tf.float32, [self.vocab_size, self.embed_size])
    word_embed = tf.nn.embedding_lookup(word_indices)

    self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
    self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

    outputs, _ = rnn.rnn(self.cell,
                         self.word_embed,
                         dtype=tf.float32)

