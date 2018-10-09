import argparse
import shutil
import time
import warnings
from datetime import datetime
import sys
import os

import collections
import gc

from torch import nn
from torch.backends import cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# import models
from training.fp16util import (model_grads_to_master_grads,
                               network_to_half,
                               prep_param_lists,
                               master_params_to_model_params)

from training import resnet
import copy

from training import dataloader
from training import experimental_utils
from training import dist_utils
from training.logger import TensorboardLogger, FileLogger
from training.meter import AverageMeter, NetworkMeter, TimeMeter


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--phases', type=str,
                        help='Specify epoch order of data resize and learning rate schedule: '
                             '[{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]')
    # parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='log/print every this many steps (default: 5)')
    parser.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode. Default True')
    parser.add_argument('--loss-scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--logdir', default='', type=str,
                        help='where logs go')
    parser.add_argument('--skip-auto-shutdown', action='store_true',
                        help='Shutdown instance at the end of training or failure')
    parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
                        help='how long to wait until shutting down on success')
    parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
                        help='how long to wait before shutting down on error')

    parser.add_argument('--short-epoch', action='store_true',
                        help='make epochs short (for debugging)')
    return parser


cudnn.benchmark = True
args = get_parser().parse_args()

# Only want master rank logging to tensorboard
is_master = (not args.distributed) or (dist_utils.env_rank() == 0)
is_rank0 = args.local_rank == 0
tb = TensorboardLogger(args.logdir, is_master=is_master)
log = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)


def main():
    os.system('shutdown -c')  # cancel previous shutdown command
    log.console(args)
    tb.log('sizes/world', dist_utils.env_world_size())

    # need to index validation directory before we start counting the time
    dataloader.sort_ar(args.data + '/validation')

    if args.distributed:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())
        log.console("Distributed: success (%d/%d)" % (args.local_rank, dist.get_world_size()))

    log.console("Loading model")
    model = resnet.resnet50(bn0=args.init_bn0).cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    best_top5 = 93  # only save models over 93%. Otherwise it stops to save every time

    global model_params, master_params
    if args.fp16:
        model_params, master_params = prep_param_lists(model)
    else:
        model_params = master_params = model.parameters()

    optim_params = experimental_utils.bnwd_optim_params(model, model_params,
                                                        master_params) if args.no_bn_wd else master_params

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(optim_params, 0, momentum=args.momentum,
                                weight_decay=args.weight_decay)  # start with 0 lr. Scheduler will change this later

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top5 = checkpoint['best_top5']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # save script so we can reproduce from logs
    shutil.copy2(os.path.realpath(__file__), f'{args.logdir}')

    log.console("Creating data loaders (this could take up to 10 minutes if volume needs to be warmed up)")
    phases = eval(args.phases)
    dm = DataManager([copy.deepcopy(p) for p in phases if 'bs' in p])
    scheduler = Scheduler(optimizer, [copy.deepcopy(p) for p in phases if 'lr' in p])

    start_time = datetime.now()  # Loading start to after everything is loaded
    if args.evaluate:
        return validate(dm.val_dl, model, criterion, 0, start_time)

    if args.distributed:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    log.event("~~epoch\thours\ttop1\ttop5\n")
    for epoch in range(args.start_epoch, scheduler.tot_epochs):
        dm.set_epoch(epoch)

        train(dm.trn_dl, model, criterion, optimizer, scheduler, epoch)
        top1, top5 = validate(dm.val_dl, model, criterion, epoch, start_time)

        time_diff = (datetime.now() - start_time).total_seconds() / 3600.0
        log.event(f'~~{epoch}\t{time_diff:.5f}\t\t{top1:.3f}\t\t{top5:.3f}\n')

        is_best = top5 > best_top5
        best_top5 = max(top5, best_top5)
        if args.local_rank == 0:
            if is_best:
                save_checkpoint(epoch, model, best_top5, optimizer, is_best=True, filename='model_best.pth.tar')
            phase = dm.get_phase(epoch)
            if phase:
                save_checkpoint(epoch, model, best_top5, optimizer, filename=f'sz{phase["bs"]}_checkpoint.path.tar')


def train(trn_loader, model, criterion, optimizer, scheduler, epoch):
    net_meter = NetworkMeter()
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(trn_loader):
        if args.short_epoch and (i > 10):
            break
        batch_num = i + 1
        timer.batch_start()
        scheduler.update_lr(epoch, i + 1, len(trn_loader))

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        if args.fp16:
            loss = loss * args.loss_scale
            model.zero_grad()
            loss.backward()
            model_grads_to_master_grads(model_params, master_params)
            for param in master_params:
                param.grad.data = param.grad.data / args.loss_scale
            optimizer.step()
            master_params_to_model_params(model_params, master_params)
            loss = loss / args.loss_scale
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train batch done. Logging results
        timer.batch_end()
        corr1, corr5 = correct(output.data, target, topk=(1, 5))
        reduced_loss, batch_total = to_python_float(loss.data), to_python_float(input.size(0))
        # Must keep track of global batch size, since not all machines are guaranteed equal batches
        # at the end of an epoch
        if args.distributed:
            metrics = torch.tensor([batch_total, reduced_loss, corr1, corr5]).float().cuda()
            batch_total, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss / dist_utils.env_world_size()
        top1acc = to_python_float(corr1) * (100.0 / batch_total)
        top5acc = to_python_float(corr5) * (100.0 / batch_total)

        losses.update(reduced_loss, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        should_print = (batch_num % args.print_freq == 0) or (batch_num == len(trn_loader))
        if args.local_rank == 0 and should_print:
            tb.log_memory()
            tb.log_trn_times(timer.batch_time.val, timer.data_time.val, input.size(0))
            tb.log_trn_loss(losses.val, top1.val, top5.val)

            recv_gbit, transmit_gbit = net_meter.update_bandwidth()
            tb.log("sizes/batch_total", batch_total)
            tb.log('net/recv_gbit', recv_gbit)
            tb.log('net/transmit_gbit', transmit_gbit)

            output = (f'Epoch: [{epoch}][{batch_num}/{len(trn_loader)}]\t'
                      f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      f'Data {timer.data_time.val:.3f} ({timer.data_time.avg:.3f})\t'
                      f'BW {recv_gbit:.3f} {transmit_gbit:.3f}')
            log.verbose(output)

        tb.update_step_count(batch_total)


def validate(val_loader, model, criterion, epoch, start_time):
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    eval_start_time = time.time()

    for i, (input, target) in enumerate(val_loader):
        if args.short_epoch and (i > 10):
            break
        batch_num = i + 1
        timer.batch_start()
        if args.distributed:
            top1acc, top5acc, loss, batch_total = distributed_predict(input, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target).data
            batch_total = input.size(0)
            top1acc, top5acc = accuracy(output.data, target, topk=(1, 5))

        # Eval batch done. Logging results
        timer.batch_end()
        losses.update(to_python_float(loss), to_python_float(batch_total))
        top1.update(to_python_float(top1acc), to_python_float(batch_total))
        top5.update(to_python_float(top5acc), to_python_float(batch_total))
        should_print = (batch_num % args.print_freq == 0) or (batch_num == len(val_loader))
        if args.local_rank == 0 and should_print:
            output = (f'Test:  [{epoch}][{batch_num}/{len(val_loader)}]\t'
                      f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            log.verbose(output)

    tb.log_eval(top1.avg, top5.avg, time.time() - eval_start_time)
    tb.log('epoch', epoch)

    return top1.avg, top5.avg


def distributed_predict(input, target, model, criterion):
    # Allows distributed prediction on uneven batches. Test set isn't always large enough for every GPU to get a batch
    batch_size = input.size(0)
    output = loss = corr1 = corr5 = valid_batches = 0

    if batch_size:
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target).data
        # measure accuracy and record loss
        valid_batches = 1
        corr1, corr5 = correct(output.data, target, topk=(1, 5))

    metrics = torch.tensor([batch_size, valid_batches, loss, corr1, corr5]).float().cuda()
    batch_total, valid_batches, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
    reduced_loss = reduced_loss / valid_batches

    top1 = corr1 * (100.0 / batch_total)
    top5 = corr5 * (100.0 / batch_total)
    return top1, top5, reduced_loss, batch_total


class DataManager():
    def __init__(self, phases):
        self.phases = self.preload_phase_data(phases)

    def set_epoch(self, epoch):
        cur_phase = self.get_phase(epoch)
        if cur_phase:
            self.set_data(cur_phase)
        if hasattr(self.trn_smp, 'set_epoch'):
            self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'):
            self.val_smp.set_epoch(epoch)

    def get_phase(self, epoch):
        return next((p for p in self.phases if p['ep'] == epoch), None)

    def set_data(self, phase):
        """Initializes data loader."""
        if phase.get('keep_dl', False):
            log.event(f'Batch size changed: {phase["bs"]}')
            tb.log_size(phase['bs'])
            self.trn_dl.update_batch_size(phase['bs'])
            return

        log.event(f'Dataset changed.'
                  f'Image size: {phase["sz"]}'
                  f'Batch size: {phase["bs"]}'
                  f'Train Directory: {phase["trndir"]}'
                  f'Validation Directory: {phase["valdir"]}')
        tb.log_size(phase['bs'], phase['sz'])

        self.trn_dl, self.val_dl, self.trn_smp, self.val_smp = phase['data']
        self.phases.remove(phase)

        # clear memory before we begin training
        gc.collect()

    def preload_phase_data(self, phases):
        for phase in phases:
            if not phase.get('keep_dl', False):
                self.expand_directories(phase)
                phase['data'] = self.preload_data(**phase)
        return phases

    def expand_directories(self, phase):
        trndir = phase.get('trndir', '')
        valdir = phase.get('valdir', trndir)
        phase['trndir'] = args.data + trndir + '/train'
        phase['valdir'] = args.data + valdir + '/validation'

    def preload_data(self, ep, sz, bs, trndir, valdir, **kwargs):  # dummy ep var to prevent error
        if 'lr' in kwargs:
            del kwargs['lr']  # in case we mix schedule and data phases
        """Pre-initializes data-loaders. Use set_data to start using it."""
        if sz == 128:
            val_bs = max(bs, 512)
        elif sz == 224:
            val_bs = max(bs, 256)
        else:
            val_bs = max(bs, 128)
        return dataloader.get_loaders(trndir, valdir, bs=bs, val_bs=val_bs, sz=sz, workers=args.workers,
                                      distributed=args.distributed, **kwargs)


# ### Learning rate scheduler
class Scheduler():
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        if len(phase['lr']) == 2:
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['ep']
        if 'epoch_step' in phase:
            batch_curr = 0  # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr
        step_size = (lr_end - lr_start) / step_tot
        return lr_start + step_curr * step_size

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if epoch >= phase['ep'][0]:
                return phase
        raise Exception('Epoch out of range')

    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1:
            return phase['lr'][0]  # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if self.current_lr == lr:
            return
        if (batch_num == 1) or (batch_num == batch_tot):
            log.event(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        tb.log("sizes/lr", lr)
        tb.log("sizes/momentum", args.momentum)


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def save_checkpoint(epoch, model, best_top5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1, 'state_dict': model.state_dict(),
        'best_top5': best_top5, 'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.logdir}/{filename}')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k."""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]


def correct(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res


def listify(p=None, q=None):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    n = q if type(q) == int else 1 if q is None else len(q)
    if len(p) == 1:
        p = p * n
    return p


if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main()
        if not args.skip_auto_shutdown:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        import traceback

        traceback.print_tb(exc_traceback, file=sys.stdout)
        log.event(e)
        # in case of exception, wait 2 hours before shutting down
        if not args.skip_auto_shutdown:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')
    tb.close()
