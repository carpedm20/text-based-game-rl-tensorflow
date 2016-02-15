import os
import numpy as np
import tensorflow as tf

from models.lstmdqn import LSTMDQN
from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("embed_dim", 100, "The dimension of word embedding matrix [100]")
flags.DEFINE_integer("seq_length", 30, "The maximum length of word [30]")
flags.DEFINE_integer("batch_size", 25, "The size of batch images [25]")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate [1.0]")
flags.DEFINE_float("decay", 0.5, "Decay of SGD [0.5]")
flags.DEFINE_string("game_name", "home", "The name of game [game]")
flags.DEFINE_string("game_dir", "game", "The name of game directory [game]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(FLAGS.checkpoint_dir)

  with tf.Session() as sess:
    model = LSTMDQN(checkpoint_dir=FLAGS.checkpoint_dir,
                    seq_length=FLAGS.seq_length,
                    embed_dim=FLAGS.embed_dim,
                    batch_size=FLAGS.batch_size,
                    forward_only=FLAGS.forward_only,
                    game_name=FLAGS.game_name,
                    game_dir=FLAGS.game_dir)

    if not FLAGS.forward_only:
      model.run()
    else:
      test_loss = model.test(2)
      print(" [*] Test loss: %2.6f, perplexity: %2.6f" % (test_loss, np.exp(test_loss)))

if __name__ == '__main__':
  tf.app.run()
