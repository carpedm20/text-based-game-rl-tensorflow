import time
from .game import Game

class FantasyGame(Game):
  def __init__(self, num_rooms=4, default_reward=-0.01,
               junk_cmd_reward=-0.1, quest_levels=1):
    super(Game, self).__init__(num_rooms, default_reward,
                               junk_cmd_reward, quest_levels)

    self.rooms = ["Living", "Garden", "Kitchen","Bedroom"]

    self.actions = ["eat", "sleep", "watch", "exercise", "go"]
    self.objects = ["north","south","east","west"]

    self.quests = ["You are hungry.","You are sleepy.", \
                   "You are bored.", "You are getting fat."]
    self.quests_mislead = ["You are not hungry.","You are not sleepy.", \
                           "You are not bored.", "You are not getting fat."]

    self.new_game()

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

    for idx in xrange(self.quest_levels):
      self.quest_checklist.append(idxs[idx])

    self.mislead_quest_checklist = idxs[-1]
    for idx in xrange(len(self.quest_checklist) - 1):
      self.mislead_quest_checklist.append(idxs[idx])

    if self.debug:
      print(" [*] Start Quest : %s %s" % (self.get_quest_text(quest_checklist[0]), self.actions[quest_checklist[0]]))
