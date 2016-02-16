from .client import TCPClient

class Game(object):
  """Game object to play
  """
  def __init__(self, num_rooms=4, default_reward=-0.01,
               junk_cmd_reward=-0.1, quest_levels=1,
               max_step=100, debug=True,
               username="root", password="root"):
    """Initialize the parameters for Game object

    Args:
      quest_levels: # of quests to complete in each run
    """
    self.num_rooms = num_rooms
    self.default_reward = default_reward
    self.junk_cmd_reward = junk_cmd_reward
    self.quest_levels = quest_levels
    self.max_step = max_step

    self.debug = debug
    self.client = TCPClient()

    self.login(username, password)

  def login(self, username, password):
    print(self.client.get())
    self.client.send("connect %s %s" % (username, password))
