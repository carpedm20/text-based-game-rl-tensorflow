import time
from .game import Game

class HomeGame(Game):
  def __init__(self, num_rooms=4,
               default_reward=-0.01,
               junk_cmd_reward=-0.1):
    super(Game, self).__init__(num_rooms, default_reward, junk_cmd_reward)

    self.rooms = ["Living", "Garden", "Kitchen","Bedroom"]

    self.actions = ["eat", "sleep", "watch", "exercise", "go"]
    self.objects = ["north","south","east","west"]

    self.quests = ["You are hungry.","You are sleepy.", \
                   "You are bored.", "You are getting fat."]
    self.quests_mislead = ["You are not hungry.","You are not sleepy.", \
                           "You are not bored.", "You are not getting fat."]

  def new_game(self):
    self.quest_checklist = []
    self.misliad_quest_checklist = []
    self.step_count = 0
    self.random_teleport()
    self.random_quest()

    return self.get_state()

  def random_teleport(self):
    room_idx = random.randrange(self.num_rooms)
    self.client.send('@tel tut#0%s' % room_idx)
    time.sleep(0.1)
    self.client.get()
    self.client.send('l')
    if self.debug:
      print(" [*] Start Room : %s %s" % (room_idx, rooms[room_idx]))

  def get_quest_text(self, quest_num):
    return self.quests_mislead[mislead_quest_checklist[0]] + " now but " + self.quests[quest_num] + " now."

  def random_quest(self):
    idxs = np.random.permutation(len(self.quests))
