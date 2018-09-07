import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os

class DDP(DistributedDataParallel):
  # Distributed wrapper. Supports asynchronous evaluation and model saving
  def forward(self, *args, **kwargs):
    # DDP has a sync point on forward. No need to do this for eval. This allows us to have different batch sizes
    if self.training: return super().forward(*args, **kwargs)
    else:             return self.module(*args, **kwargs)

  def load_state_dict(self, *args, **kwargs):
    self.module.load_state_dict(*args, **kwargs)

  def state_dict(self, *args, **kwargs):
    return self.module.state_dict(*args, **kwargs)
    



def reduce_tensor(tensor): return sum_tensor(tensor)/env_world_size()
def sum_tensor(tensor):
  rt = tensor.clone()
  dist.all_reduce(rt, op=dist.reduce_op.SUM)
  return rt

def env_world_size(): return int(os.environ['WORLD_SIZE'])
def env_rank(): return int(os.environ['RANK'])

