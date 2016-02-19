import re
import telnetlib

ansi_escape = re.compile(r'\x1b[^m]*m')

class TCPClient(object):
  def __init__(self, host="localhost", port=4001, timeout=0.002):
    self.host = host
    self.port = port
    self.timeout = timeout

    self.connect()

  def connect(self):
    self.client = telnetlib.Telnet(self.host, self.port, self.timeout)

  def get(self, count=1, timeout=False):
    for _ in xrange(count):
      if timeout:
        data = self.client.read_until('<EOM>', self.timeout)[:-5]
      else:
        data = self.client.read_until('<EOM>')[:-5]

      if False:
        print("***********get***************")
        print(data)
        print("*****************************")
    return ansi_escape.sub('', data).strip()

  def send(self, data):
    if False:
      print("===========send==============")
      print(data)
      print("=============================")
    self.client.write(data + "\n")
