import getpass
import sys
import telnetlib

class Game(object):
  def __init__(self, host="localhost", start_port=4000,
               user="root", password="root", num_game=4):
    self.host = host
    self.start_port = start_port
    sefl.user = user
    self.password = password

    self.num_game = num_game

  def forget(self):
    pass

  def perceive(self):
    pass
