from typing import Optional
from torch.optim.optimizer import Optimizer
import sys
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torchvision import transforms
from torch.optim.optimizer import Optimizer, required
import copy
import torch.optim

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])


class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

# class newSGD(Optimizer):

#     def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay1=0, tau=0,
#                  weight_decay=0, nesterov=False):
#         global pi_p
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if tau < 0.0:
#             raise ValueError("Invalid noisy rate: {}".format(tau))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, weight_decay1=weight_decay1,
#                         tau = tau, nesterov=nesterov)
#         super(newSGD, self).__init__(params, defaults)
        
#         # for group in self.param_groups:
#         data1 = copy.deepcopy(self.param_groups)
#         pi  = []
#         pi_p = []
#         for group in data1:
#             pi += group['params']
#         for index in range(len(pi)):
#             pi_p.append(pi[index].data)

#     def __setstate__(self, state):
#         super(newSGD, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)

#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#         index = 0
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']
#             tau = group['tau']

#             for p in group['params']:
#                 p_init = pi_p[index]
#                 index += 1
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if tau > 0:
#                     g = (p + p_init).abs()
#                     m = p.numel()
#                     mn = int(m*tau) 
#                     if mn>0:
#                         kth,_ = g.cpu().flatten().kthvalue(mn)
#                         d_p = torch.where(g < kth.cuda(),d_p.mul(0).add(torch.sign(p.data), alpha=weight_decay), d_p.add(p.data, alpha=weight_decay))       
#                 elif weight_decay != 0:
#                     d_p.add_(weight_decay, p.data)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
#                         buf.mul_(momentum).add_(d_p)
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(1 - dampening, d_p)
#                     if nesterov:
#                         d_p = d_p.add(momentum, buf)
#                     else:
#                         d_p = buf

#                 p.data.add_(-group['lr'], d_p)

#         return loss
class newSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,tau=0):
        global pi_p
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(newSGD, self).__init__(params, defaults)
        data1 = copy.deepcopy(self.param_groups)
        pi  = []
        pi_p = []
        for group in data1:
            pi += group['params']
        for index in range(len(pi)):
            pi_p.append(pi[index].data)

    def __setstate__(self, state):
        super(newSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        global pi_p
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        index = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            tau = group['tau']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if tau > 0 :
                    p_init = pi_p[index]
                    index += 1
                    g = (p + p_init).abs()
                    m = p.numel()
                    mn = int(m*tau) 
                    if mn>0:
                        kth,_ = g.cpu().flatten().kthvalue(mn)
                        d_p = torch.where(g < kth.cuda(),d_p.mul(0).add(p.data, alpha=weight_decay*5), d_p.add(p.data, alpha=weight_decay))      
                elif weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


# class newSGD(Optimizer):
#     #存一下初始化的参数
    
   
#     def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay1=0, tau=0,
#                  weight_decay=0, nesterov=False):
#         global pi_p
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if tau < 0.0:
#             raise ValueError("Invalid noisy rate: {}".format(tau))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, weight_decay1=weight_decay1,
#                         tau = tau, nesterov=nesterov)
#         super(newSGD, self).__init__(params, defaults)
        
#         # for group in self.param_groups:
#         data1 = copy.deepcopy(self.param_groups)
#         pi  = []
#         pi_p = []
#         for group in data1:
#             pi += group['params']
#         for index in range(len(pi)):
#             pi_p.append(pi[index].data)

#     def __setstate__(self, state):
#         super(newSGD, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)


#     def step(self, closure=None):
#         global pi_p
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#         index =0
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']
#             tau = group['tau']
#             for p in group['params']:
#                 p_init = pi_p[index]
#                 index += 1
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if tau > 0:
#                     g = (p + p_init).abs()
#                     m = p.numel()
#                     mn = int(m*tau) 
#                     if mn>0:
#                         kth,_ = g.cpu().flatten().kthvalue(mn)
#                         d_p = torch.where(g < kth.cuda(),d_p.mul(0).add(torch.sign(p.data), alpha=weight_decay), d_p.add(p.data, alpha=weight_decay))       
#                 elif weight_decay != 0:
#                     d_p = d_p.add(p.data, alpha=weight_decay)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
#                         buf.mul_(momentum).add_(d_p)
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(1 - dampening, d_p)
#                     if nesterov:
#                         d_p = d_p.add(momentum, buf)
#                     else:
#                         d_p = buf

#                 p.data.add_(-group['lr'], d_p)

#         return loss

class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class AccuracyCounter:

    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)



def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    
    return entropy





def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x
