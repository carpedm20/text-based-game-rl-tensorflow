import getpass
import sys
import telnetlib

class Agent(object):
  def __init__(self, host="localhost", start_port=4000,
               user="root", password="root", name="home",
               num_game=4):
    self.host = host
    self.start_port = start_port
    sefl.user = user
    self.password = password

    self.num_game = num_game

    if name == "home":
      self.action = 4
    elif name == "fantasy":
      self.action = 4
    else:
      raise Exception(" [!] Wrong game name : %s" % name)

  def forget(self):
    pass

  def do(self, action):
    state, reward, is_end = None, None, None

    return state, reward, is_end
