import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from .base import Model

class LSTMDQN(Model):
  """LSTM Deep Q Network
  """
  def __init__(self, rnn_size=100, batch_size=25,
               seq_length=30, embed_dim=100, layer_depth=3,
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
    self.layer_depth = layer_depth

    self.embed_dim = embed_dim
    self.vocab_size = 100

    self.game_dir = game_dir
    self.game_name = game_name

    self.build_model()

  def build_model(self):
    #self.inputs = tf.placeholder(tf.int32, [None, self.seq_length])
    self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

    embed = tf.get_variable("embed", [self.vocab_size, self.embed_dim])
    word_embeds = tf.nn.embedding_lookup(embed, self.inputs)

    self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
    self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

    outputs, _ = rnn.rnn(self.cell,
                         [tf.squeeze(embed_t) for embed_t in tf.split(1, self.seq_length, word_embeds)],
                         dtype=tf.float32)

    output_embed = tf.pack(outputs)
    mean_pool = tf.nn.relu(tf.reduce_mean(output_embed, 1))

    self.action_size = 4
    self.object_size = 4

    action = rnn_cell.linear(mean_pool, self.action_size, 0, "action")
    object_ = rnn_cell.linear(mean_pool, self.object_size, 0, "object")

  def run(self):
    pass
