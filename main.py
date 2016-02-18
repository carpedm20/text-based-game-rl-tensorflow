import os
import numpy as np
import tensorflow as tf

from games import HomeGame, FantasyGame
from models.lstmdqn import LSTMDQN

from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("embed_dim", 100, "The dimension of word embedding matrix [100]")
flags.DEFINE_integer("seq_length", 30, "The maximum length of word [30]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [1]")
flags.DEFINE_integer("layer_depth", 1, "The size of batch images [1]")
flags.DEFINE_integer("epsilon_end_time", 1000000, "# of time step to decay epsilon [1000000]")
flags.DEFINE_float("start_epsilon", 1.0, "The start value of epsilon [1.0]")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate [1.0]")
flags.DEFINE_float("decay", 0.5, "Decay of SGD [0.5]")
flags.DEFINE_string("game_name", "home", "The name of game [game]")
flags.DEFINE_string("game_dir", "../text-world", "The name of game directory [game]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(FLAGS.checkpoint_dir)

  if FLAGS.game_name == "home":
    game = HomeGame(game_dir=FLAGS.game_dir, seq_length=FLAGS.seq_length)
  else:
    raise Exception(" [!] %s not implemented yet" % self.game_name)

  with tf.device('/cpu:0'):
    model = LSTMDQN(game, checkpoint_dir=FLAGS.checkpoint_dir,
                    seq_length=FLAGS.seq_length,
                    embed_dim=FLAGS.embed_dim,
                    layer_depth=FLAGS.layer_depth,
                    batch_size=FLAGS.batch_size,
                    start_epsilon=FLAGS.start_epsilon,
                    forward_only=FLAGS.forward_only)

    if not FLAGS.forward_only:
      model.train()
    else:
      test_loss = model.test(2)
      print(" [*] Test loss: %2.6f, perplexity: %2.6f" % (test_loss, np.exp(test_loss)))

if __name__ == '__main__':
  tf.app.run()
