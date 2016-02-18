import time
import random
import numpy as np
import tensorflow as tf
from collections import deque

from .base import Model

class LSTMDQN(Model):
  """LSTM Deep Q Network
  """
  def __init__(self, game, rnn_size=100, batch_size=25,
               seq_length=30, embed_dim=100, layer_depth=3,
               start_epsilon=1, epsilon_end_time=1000000,
               memory_size=1000000, 
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
    self.final_epsilon = 0.05
    self.observe = 500
    self.explore = 500
    self.gamma = 0.99
    self.num_action_per_step = 1
    self.memory_size = memory_size

    self.game = game
    self.dataset = game.name

    self.num_action = len(self.game.actions)
    self.num_objects = len(self.game.objects)

    self._attrs = ['epsilon', 'final_epsilon', 'oberve', \
        'explore', 'gamma', 'memory_size', 'batch_size']

    self.build_model()

  def build_model(self):
    # Representation Generator
    self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

    embed = tf.get_variable("embed", [self.vocab_size, self.embed_dim])
    word_embeds = tf.nn.embedding_lookup(embed, self.inputs)

    self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
    self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

    outputs, _ = tf.nn.rnn(self.cell,
        [tf.squeeze(embed_t) for embed_t in tf.split(1, self.seq_length, word_embeds)],
                            dtype=tf.float32)

    output_embed = tf.transpose(tf.pack(outputs), [1, 0, 2])
    mean_pool = tf.nn.relu(tf.reduce_mean(output_embed, 1))

    # Action scorer. no bias in paper
    self.pred_action = tf.nn.rnn_cell.linear(mean_pool, self.num_action, 0.0, scope="action")
    self.pred_object = tf.nn.rnn_cell.linear(mean_pool, self.num_objects, 0.0, scope="object")

    self.true_action = tf.placeholder(tf.float32, [self.batch_size, self.num_action])

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
    with self.sess:
      self.max_iter = max_iter
      self.alpha = alpha
      self.learning_rate = learning_rate
      self.checkpoint_dir = checkpoint_dir

      self.step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_sum(tf.square(self.true_action - self.pred_action))
      _ = tf.scalar_summary("loss", self.loss)

      self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

      self.memory = deque()

      action = np.zeros(self.num_action)
      action[0] = 1

      self.initialize(log_dir="./logs")

      start_time = time.time()
      start_iter = self.step.eval()

      state_t, reward, is_finished = self.game.new_game()
      state_t = np.tile(state_t, [self.batch_size,1])

      for step in xrange(start_iter, start_iter + self.max_iter): 
        pred_action, pred_object = self.sess.run(
            [self.pred_action, self.pred_object], feed_dict={self.inputs: state_t})

        action_t = np.zeros([self.num_action])
        object_t = np.zeros([self.num_object])

        # Epsilon greedy
        if random.random() <= self.epsilon or step <= observe:
          action_idx = random.randrange(0, self.num_action - 1)
          object_idx = random.randrange(0, self.num_action - 1)
        else:
          action_idx = np.argmax(pred_action)
          object_idx = np.argmax(pred_object)

        action_t[action_idx] = 1
        object_t[object_idx] = 1

        if self.epsilon > self.final_epsilon and step > self.observe:
          self.epsilon -= (self.initial_epsilon- self.final_epsilon) / self.observe

        # run and observe rewards
        for idx in xrange(self.num_action_per_step):
          # evaluate all other actions
          for action in len(game.actions):
            pass
            game.do(action)
          import ipdb; ipdb.set_trace() 

        if step > self.observe:
          batch = random.sample(memory, self.batch_size)

          s = [mem[0] for mem in batch]
          a = [mem[1] for mem in batch]
          o = [mem[2] for mem in batch]
          r = [mem[3] for mem in batch]
          s2 = [mem[4] for mem in batch]
          term = [mem[5] for mem in batch]
          avail_objects = [mem[6] for mem in batch]

          y_batch = []
          action = pred_action.eval(feed_dict={self.inputs: s})
          for idx in xrange(self.batch_size):
            if batch[idx][4]:
              y_batch.append(r[idx])
            else:
              y_batch.append(r[idx] + self.gamma * np.max(action[idx]))

          train.run(feed_dict={
            true_action: None,
            pred_action: None,
            s: None
          })

        if terminal:
          state_t, reward, is_finished = self.game.new_game()
        else:
          state_t, reward, is_finished = self.game.get_state()

        state_t = state_t1
