import sys
import time
import socket

class TCPClient(object):
  def __init__(self, host="localhost", port=4001, timeout=0.001):
    self.host = host
    self.port = port
    self.timeout = timeout

    self.connect()

  def connect(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect((self.host, self.port))
    self.socket.settimeout(self.timeout)
    self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

  def get(self):
    msg = ""
    while True:
      try:
        data = self.socket.recv(1024)
      except:
        break
      msg += data
    time.sleep(0.0005)
    return msg.split('\n')

  def send(self, data):
    self.socket.send(data + "\n")
