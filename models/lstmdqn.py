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
    self.num_object = len(self.game.objects)

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
        [tf.reshape(embed_t, [self.batch_size, self.embed_dim]) for embed_t in tf.split(1, self.seq_length, word_embeds)],
                            dtype=tf.float32)

    output_embed = tf.transpose(tf.pack(outputs), [1, 0, 2])
    mean_pool = tf.nn.relu(tf.reduce_mean(output_embed, 1))

    # Action scorer. no bias in paper
    self.pred_reward = tf.nn.rnn_cell.linear(mean_pool, self.num_action, 0.0, scope="action")
    self.pred_object = tf.nn.rnn_cell.linear(mean_pool, self.num_object, 0.0, scope="object")

    self.true_reward = tf.placeholder(tf.float32, [self.batch_size, self.num_action])
    self.true_object = tf.placeholder(tf.float32, [self.batch_size, self.num_object])

    _ = tf.histogram_summary("mean_pool", mean_pool)
    _ = tf.histogram_summary("pred_reward", self.pred_reward)
    _ = tf.histogram_summary("true_reward", self.true_reward)

    _ = tf.scalar_summary("pred_reward_mean", tf.reduce_mean(self.pred_reward))
    _ = tf.scalar_summary("true_reward_mean", tf.reduce_mean(self.true_reward))

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

      self.loss = tf.reduce_sum(tf.square(self.true_reward - self.pred_reward))
      self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

      _ = tf.scalar_summary("loss", self.loss)

      self.memory = deque()

      action = np.zeros(self.num_action)
      action[0] = 1

      self.initialize(log_dir="./logs")

      start_time = time.time()
      start_iter = self.step.eval()

      state_t, reward, is_finished = self.game.new_game()

      win_count = 0
      steps = xrange(start_iter, start_iter + self.max_iter)
      print(" [*] Start")

      for step in steps:
        pred_reward, pred_object = self.sess.run(
            [self.pred_reward, self.pred_object], feed_dict={self.inputs: [state_t]})

        action_t = np.zeros([self.num_action])
        object_t = np.zeros([self.num_object])

        # Epsilon greedy
        if random.random() <= self.epsilon or step <= self.observe:
          action_idx = random.randrange(0, self.num_action - 1)
          object_idx = random.randrange(0, self.num_action - 1)
        else:
          max_reward = np.max(pred_reward[0])
          max_object = np.max(pred_object[0])

          action_idx = np.random.choice(np.where(pred_reward[0] == max_reward)[0])
          object_idx = np.random.choice(np.where(pred_object[0] == max_object)[0])
          #best_q = (max_action + max_object)/2

        # run and observe rewards
        action_t[action_idx] = 1
        object_t[object_idx] = 1

        if self.epsilon > self.final_epsilon and step > self.observe:
          self.epsilon -= (self.start_epsilon- self.final_epsilon) / self.observe

        state_t1, reward_t, is_finished = self.game.do(action_idx, object_idx)
        self.memory.append((state_t, action_t, object_t, reward_t, state_t1, is_finished))

        # qLearnMinibatch : Q-learning updates
        if step > self.observe:
          batch = random.sample(self.memory, self.batch_size)

          s = [mem[0] for mem in batch]
          a = [mem[1] for mem in batch]
          o = [mem[2] for mem in batch]
          r = [mem[3] for mem in batch]
          s2 = [mem[4] for mem in batch]
          finished = [mem[5] for mem in batch]

          if r > 0:
            win_count += 1

          pred_reward = self.pred_reward.eval(feed_dict={self.inputs: s2})

          action = np.zeros(self.num_action)
          object_= np.zeros(self.num_object)

          _, loss, summary_str = self.sess.run([self.optim, self.loss, self.merged_sum], feed_dict={
            self.inputs: s,
            self.true_reward: a,
            self.pred_reward: pred_reward,
            self.true_object: o,
            self.pred_object: pred_object,
          })

          if step % 10000 == 0:
            self.save(checkpoint_dir, step)

          if step % 50 == 0:
            print("Step: [%2d/%7d] time: %4.4f, loss: %.8f, win: %4d" % (step, self.max_iter, time.time() - start_time, loss, win_count))

        if is_finished:
          state_t, reward, is_finished = self.game.new_game()

        state_t = state_t1
