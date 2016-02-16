from ..client import TCPClient

class Game(object):
  def __init__(self, num_rooms=4, default_reward=-0.01,
               junk_cmd_reward=-0.1):
    self.num_rooms = num_rooms
    self.default_reward = default_reward
    self.junk_cmd_reward = junk_cmd_reward

    self.client = TCPClient()
