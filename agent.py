from game import HomeGame, FantasyGame

class Agent(object):
  def __init__(self, host="localhost", start_port=4000,
               user="root", password="root", name="home",
               num_game=4, debug=True):
    self.host = host
    self.start_port = start_port
    sefl.user = user
    self.password = password

    self.num_game = num_game
    self.debug = debug

    if name == "home":
      game = HomeGame()
    elif name == "fantasy":
      game = FantasyGame()
    else:
      raise Exception(" [!] Wrong game name : %s" % name)

  def forget(self):
    pass

  def do(self, action):
    state, reward, is_end = None, None, None

    return state, reward, is_end
