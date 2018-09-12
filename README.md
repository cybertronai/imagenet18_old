Code to reproduce ImageNet in 18 minutes, by Andrew Shaw, Yaroslav Bulatov, and Jeremy Howard.


Pre-requisites: Python 3.6

```
pip install -r requirements.txt
aws configure  (or set your AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_DEFAULT_REGION)
python train.py  # pre-warming
python train.py 
```

To run with smaller number of machines:

```
python train.py --machines=1
python train.py --machines=4
python train.py --machines=8
python train.py --machines=16
```

# Checking progress

Machines print progress to local stdout as well as logging TensorBoard event files to EFS. You can:

1. launch tensorboard using tools/launch_tensorboard.py

That will provide a link to tensorboard instance which has loss graph under "losses" group. You'll see something like this under "Losses" tab
<img src='https://raw.githubusercontent.com/diux-dev/imagenet18/master/tensorboard.png'>

2. Connect to one of the instances using instructions printed during launch. Look for something like this

```
2018-09-06 17:26:23.562096 15.imagenet: To connect to 15.imagenet
ssh -i /Users/yaroslav/.ncluster/ncluster5-yaroslav-316880547378-us-east-1.pem -o StrictHostKeyChecking=no ubuntu@18.206.193.26
tmux a
```

This will connect you to tmux session and you will see something like this

```
.997 (65.102)   Acc@5 85.854 (85.224)   Data 0.004 (0.035)      BW 2.444 2.445
Epoch: [21][175/179]    Time 0.318 (0.368)      Loss 1.4276 (1.4767)    Acc@1 66.169 (65.132)   Acc@5 86.063 (85.244)   Data 0.004 (0.035)      BW 2.464 2.466
Changing LR from 0.4012569832402235 to 0.40000000000000013
Epoch: [21][179/179]    Time 0.336 (0.367)      Loss 1.4457 (1.4761)    Acc@1 65.473 (65.152)   Acc@5 86.061 (85.252)   Data 0.004 (0.034)      BW 2.393 2.397
Test:  [21][5/7]        Time 0.106 (0.563)      Loss 1.3254 (1.3187)    Acc@1 67.508 (67.693)   Acc@5 88.644 (88.315)
Test:  [21][7/7]        Time 0.105 (0.432)      Loss 1.4089 (1.3346)    Acc@1 67.134 (67.462)   Acc@5 87.257 (88.124)
~~21    0.31132         67.462          88.124
```

The last number indicates that at epoch 21 the run got 67.462 top-1 test accuracy and 88.124 top-5 test accuracy.
