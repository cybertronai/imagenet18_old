
import subprocess, time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0 # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = val if self.count == 0 else self.avg*self.avg_mom + val*(1-self.avg_mom)
        self.avg = self.sum / self.count

class NetworkMeter:
  def __init__(self):
    self.recv_meter = AverageMeter()
    self.transmit_meter = AverageMeter()
    self.last_recv_bytes, self.last_transmit_bytes = network_bytes()
    self.last_log_time = time.time()

  def update_bandwidth(self):
    time_delta = time.time()-self.last_log_time
    recv_bytes, transmit_bytes = network_bytes()
    
    recv_delta = recv_bytes - self.last_recv_bytes
    transmit_delta = transmit_bytes - self.last_transmit_bytes

    # turn into Gbps
    recv_gbit = 8*recv_delta/time_delta/1e9
    transmit_gbit = 8*transmit_delta/time_delta/1e9
    self.recv_meter.update(recv_gbit)
    self.transmit_meter.update(transmit_gbit)
    
    self.last_log_time = time.time()
    self.last_recv_bytes = recv_bytes
    self.last_transmit_bytes = transmit_bytes
    return recv_gbit, transmit_gbit

class TimeMeter:
  def __init__(self):
    self.batch_time = AverageMeter()
    self.data_time = AverageMeter()
    self.start = time.time()

  def batch_start(self):
    self.data_time.update(time.time() - self.start)

  def batch_end(self):
    self.batch_time.update(time.time() - self.start)
    self.start = time.time()

      
################################################################################
# Generic utility methods, eventually refactor into separate file
################################################################################
def network_bytes():
  """Returns received bytes, transmitted bytes."""
  
  proc = subprocess.Popen(['cat', '/proc/net/dev'], stdout=subprocess.PIPE)
  stdout,stderr = proc.communicate()
  stdout=stdout.decode('ascii')

  recv_bytes = 0
  transmit_bytes = 0
  lines=stdout.strip().split('\n')
  lines = lines[2:]  # strip header
  for line in lines:
    line = line.strip()
    # ignore loopback interface
    if line.startswith('lo'):
      continue
    toks = line.split()

    recv_bytes += int(toks[1])
    transmit_bytes += int(toks[9])
  return recv_bytes, transmit_bytes

################################################################################
