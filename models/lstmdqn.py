import tensorflow as tf

from .base import Model

class LSTMDQN(Model):
  """LSTM Deep Q Network
  """
  def __init__(self, rnn_size=100, batch_size=25,
               seq_length=30, embed_dim=100,
               game_dir="enivia", game_name="home",
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for LSTM DQN

    Args:
      rnn_size: the dimensionality of hidden layers
      layer_depth: # of depth in LSTM layers
      batch_size: size of batch per epoch
      embed_dim: the dimensionality of word embeddings
    """
    self.sess = tf.Session()

    self.rnn_size = rnn_size
    self.seq_length = seq_length
    self.batch_size = batch_size

    self.embed_dim = embed_dim
    self.vocab_size = 100

    self.game_dir = game_dir
    self.game_name = game_name

    self.build_model()

  def build_model(self):
    self.inputs = tf.placeholder(tf.float32, [None, self.seq_length])
    word_indices = tf.split(1, self.seq_length, tf.expand_dims(self.inputs, -1))

    embed = tf.get_variable(tf.float32, [self.vocab_size, self.embed_dim])
    word_embeds = [tf.nn.embedding_lookup(word_index) for word_index in word_indices]

    self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
    self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

    outputs, _ = rnn.rnn(self.cell,
                         self.word_embed,
                         dtype=tf.float32)

    import ipdb; ipdb.set_trace() 
    tmp = 123

  def run(self):
    pass
