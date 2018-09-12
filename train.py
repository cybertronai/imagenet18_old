#!/usr/bin/env python

import argparse
import ncluster
import os

IMAGE_NAME = 'pytorch.imagenet.source.v7'
INSTANCE_TYPE = 'p3.16xlarge'
NUM_GPUS = 8

ncluster.set_backend('aws')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='imagenet',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=16,
                    help="how many machines to use")
args = parser.parse_args()

# 109:12 to 93.00
# events: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-1
# logs: https://s3.amazonaws.com/yaroslavvb/logs/imagenet1.tar
lr = 1.0
scale_224 = 224/512
scale_288 = 128/512
one_machine = [
  {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
  {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
  {'ep':5, 'lr':lr},
  {'ep':14, 'sz':224, 'bs':224,
                'lr':lr*scale_224},
  {'ep':16,     'lr':lr/10*scale_224},
  {'ep':27,     'lr':lr/100*scale_224},
  {'ep':32, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True,
                'lr':lr/100*scale_288},
  {'ep':(33,35),'lr':lr/1000*scale_288}
]

# 29:44 to 93.05
# events: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-4
# logs: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-4.tar
lr = 0.50 * 4 # 4 = num tasks
bs = [256, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs] # scale learning rate to batch size
four_machines = [
  {'ep':0,  'sz':128, 'bs':bs[0], 'trndir':'-sz/160'}, # bs = 256 * 4 * 8 = 8192
  {'ep':(0,6),  'lr':(lr,lr*2)}, 
  {'ep':6,  'sz':128, 'bs':bs[0]*2, 'keep_dl':True},
  {'ep':6,      'lr':lr*2},
  {'ep':(11,13), 'lr':(lr*2,lr)}, # trying one cycle
  {'ep':13, 'sz':224, 'bs':bs[1], 'trndir': '-sz/352', 'min_scale': 0.087},
  {'ep':13,     'lr':lr*bs_scale[1]},
  {'ep':(16,23),'lr':(lr*bs_scale[1],lr/10*bs_scale[1])},
  {'ep':(23,28),'lr':(lr/10*bs_scale[1],lr/100*bs_scale[1])},
  {'ep':28, 'sz':288, 'bs':bs[2], 'min_scale':0.5, 'rect_val':True},
  {'ep':(28,30),'lr':(lr/100*bs_scale[2],lr/1000*bs_scale[2])}
]


# 19:04 to 93.0
# events: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-16.02.8
# logs: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-8.tar 
lr = 0.24 * 8
scale_224 = 224/128
eight_machines = [
  {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
  {'ep':(0,6),  'lr':(lr,lr*2)},
  {'ep':6,            'bs':256, 'keep_dl':True,
                'lr':lr*2},
  {'ep':(11,14),'lr':(lr*2,lr)}, # trying one cycle
  {'ep':14, 'sz':224, 'bs':128, 'trndir':'-sz/352', 'min_scale':0.087,
                'lr':lr},
  {'ep':17,           'bs':224, 'keep_dl':True},
  {'ep':(17,23),'lr':(lr,lr/10*scale_224)},
  {'ep':(23,29),'lr':(lr/10*scale_224,lr/100*scale_224)},
  {'ep':29, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
  {'ep':(29,35),'lr':(lr/100,lr/1000)}
]

# 16:08 to 93.04 (after prewarming)
# events: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-16.02.thu16
# logs: https://s3.amazonaws.com/yaroslavvb/logs/imagenet-16.cmd.tar
lr = 0.235 * 8 # 
bs = 64
sixteen_machines = [
    {'ep':0,  'sz':128, 'bs':64, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':128, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':16, 'sz':224,'bs':64}, # todo: increase this bs
    {'ep':16,      'lr':lr},
    {'ep':19,           'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':2*lr/(10/1.5)},
    {'ep':31,     'lr':2*lr/(100/1.5)},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
    {'ep':37,     'lr':2*lr/100},
    {'ep':(38,50),'lr':2*lr/1000}
]
  
schedules = {1: one_machine,
             4: four_machines,
             8: eight_machines,
             16: sixteen_machines}


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
  if num_tasks <= 1:
    return 'NCCL_DEBUG=VERSION'
  nccl_rings = get_nccl_rings(num_tasks, num_gpus)
  return f'NCCL_RINGS="{nccl_rings}" NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'
  # return 'NCCL_MIN_NRINGS=2 NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'


def get_nccl_rings(num_tasks, num_gpus):
  ring = build_ring_order(range(num_tasks), range(num_gpus))
  ring_rev = build_ring_order(reversed(range(num_tasks)),
                              reversed(range(num_gpus)))
  rotated_gpu_order = [3, 2, 1, 0, 7, 6, 5, 4]
  skip_gpu_order = get_skip_order(num_gpus)
  if (num_tasks >= 4) and (num_gpus == 8):
    assert ((num_tasks % 4) == 0)
    skip_machine_order = get_skip_order(num_tasks)
    ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
    ring_skip_rev = build_ring_order(reversed(skip_machine_order),
                                     skip_gpu_order)
    rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
    # rings_arr = [ring, ring_rev, ring_skip]
  else:
    rings_arr = [ring, ring_rev]
  return ' | '.join(rings_arr)


def build_ring_order(machine_order, gpu_order):
  gpu_order = list(gpu_order)
  machine_order = list(machine_order)
  ngpus = len(gpu_order)
  r_order = [(x * ngpus) + y for x in machine_order for y in gpu_order]
  return ' '.join(map(str, r_order))


def get_skip_order(size):
  if size == 4:
    return [0, 2, 1, 3]
  skip_step = 5 if size == 16 else 3
  # step size of 3 yields - [0,3,6,1,4,7,2,5]
  return [(i * skip_step) % size for i in range(size)]


def format_params(arg):
  if isinstance(arg, list) or isinstance(arg, dict):
    return '\"' + str(arg) + '\"'
  else:
    return str(arg)


def main():
  supported_regions = ['us-west-2', 'us-east-1', 'us-east-2']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"
  assert args.machines in schedules, f"{args.machines} not supported, only support {schedules.keys()}"

  os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'  # use io2 disk on AWS
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                          install_script=open('setup.sh').read())
  job.upload('training')
  job.run(f'source activate pytorch_source')

  nccl_params = get_nccl_params(args.machines, NUM_GPUS)

  # Training script args
  default_params = [
    '~/data/imagenet',
    '--fp16',
    '--logdir', job.logdir,
    '--distributed',
    '--init-bn0',
    '--no-bn-wd',
  ]

  params = ['--phases', schedules[args.machines]]
  training_params = default_params + params
  training_params = ' '.join(map(format_params, training_params))

  # TODO: simplify args processing, or give link to actual commands run
  for i, task in enumerate(job.tasks):
    dist_params = f'--nproc_per_node=8 --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
    cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} training/train_imagenet_nv.py {training_params}'
    task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
    task.run(cmd, non_blocking=True)

  print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
