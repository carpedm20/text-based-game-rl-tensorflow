from .client import TCPClient

class Game(object):
  """Game object to play
  """
  def __init__(self, num_rooms=4, default_reward=-0.01,
               junk_cmd_reward=-0.1, quest_levels=1,
               seq_length=100, max_step=100, debug=True,
               username="root", password="root",
               game_dir="home"):
    """Initialize the parameters for Game object

    Args:
      quest_levels: # of quests to complete in each run
    """
    self.num_rooms = int(num_rooms)
    self.default_reward = float(default_reward)
    self.junk_cmd_reward = float(junk_cmd_reward)
    self.quest_levels = int(quest_levels)
    self.max_step = int(max_step)
    self.seq_length = int(seq_length)

    self.debug = debug
    self.client = TCPClient()
    self.client.get()

    self.game_dir = game_dir
    self.login(username, password)

  def login(self, username, password):
    self.client.send("connect %s %s" % (username, password))
    self.client.get(3)
