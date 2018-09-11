#!/usr/bin/env python
# Usage:
# ./launch_tensorboard.py
#
# This will launch r5.large machine on AWS with tensoboard, and print URL
# in the console
import ncluster
ncluster.use_aws()

task = ncluster.make_task('tensorboard',
                          instance_type='r5.large',
                          image_name='Deep Learning AMI (Ubuntu) Version 13.0')
task.run('source activate tensorflow_p36')
task.run(f'tensorboard --logdir={task.logdir}/..', non_blocking=True)
print(f"Tensorboard at http://{task.public_ip}:6006")
