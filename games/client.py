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
    self.client = telnetlib.Telnet(self.host, self.port)

  def get(self):
    data = self.client.read_until('<EOM>')[:-5]
    return ansi_escape.sub('', data).strip()

  def send(self, data):
    print("=============================")
    print(data)
    print("=============================")
    self.client.write(data + "\n")
