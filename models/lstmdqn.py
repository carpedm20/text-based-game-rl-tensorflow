import time
import random
import tensorflow as tf
from collections import deque
from tensorflow.models.rnn import rnn, rnn_cell

from .base import Model
from ..game import Agent

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

    self.epsilon = self.start_epsilon = start_epsilon
    self.final_epsilon = final_epsilon
    self.memory_size = memory_size

    self.game_dir = game_dir
    self.game_name = game_name

    self.game = Agent(self.game_dir, self.game_name)
    self.action = self.game.action
    self.object_ = self.game.object_

    self.build_model()

  def build_model(self):
    # Representation Generator
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

    self.num_action = 4
    self.object_size = 4

    # Action scorer. no bias in paper
    self.pred_action = rnn_cell.linear(mean_pool, self.num_action, 0, "action")
    self.object_ = rnn_cell.linear(mean_pool, self.object_size, 0, "object")

    self.true_action = tf.placeholder(tf.int32, [self.batch_size, self.num_action])

  def train(self, max_iter=1000000,
            alpha=0.01, learning_rate=0.001,
            start_epsilon=1.0, final_epsilon=0.05, memory_size=5000,
            checkpoint_dir="checkpoint"):
    """Train an LSTM Deep Q Network.

    Args:
      max_iter: int, The size of total iterations [450000]
      alpha: float, The importance of regularizer term [0.01]
      learning_rate: float, The learning rate of SGD [0.001]
      checkpoint_dir: str, The path for checkpoints to be saved [checkpoint]
    """
    self.max_iter = max_iter
    self.alpha = alpha
    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.step = tf.Variable(0, trainable=False)

    self.loss = tf.reduce_sum(tf.square(self.true_action - self.pred_action))
    _ = tf.scalar_summary("loss", self.loss)

    self.optim = self.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    self.memory = deque()

    action = np.zeros(self.num_action)
    action[0] = 1

    sentences_t, reward, is_end = agent.do(action)
    state_t = np.stack((sentences_t, sentences_t, sentences_t, sentences_t), axis=2)

    self.initialize(log_dir="./logs")

    start_time = time.time()
    start_iter = self.step.eval()

    for step in xrange(start_iter, start_iter + self.max_iter): 
      otuput_t = self.pred_action.eval(feed_dict = {self.inputs: state_t})
      action_t = np.zeros([self.num_action])

      if random.random() <= self.epsilon or step <= observe:
        action_idx = random.randrange(0, self.num_action - 1)
      else:
        action_idx = np.argmax(output_t)

      action_t[action_idx] = 1

      if epsilon > final_epsilon and t > observe:
        epsilon -= (self.start_epsilon - self.final_epsilon) / self.explore_t

