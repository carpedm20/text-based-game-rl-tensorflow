import sys
import string
import pprint

try:
    xrange
except NameError:
    xrange = range

pp = pprint.PrettyPrinter()

def clean_words(words):
  return [word.lower().translate(None, string.punctuation) for word in self.idx2word]

