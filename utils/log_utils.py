import os
import sys
import time
import logging
from tensorboardX import SummaryWriter
from collections import OrderedDict

def pprint_args(args):
    string = []
    string.append("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        string.append('\t{}: {}'.format(p, v))
    string = '\n'.join(string)
    return string

def init_logging(args):
    """ This logger copy all code, don't put too many things in code folder
        Initializes the logging and the SummaryWriter modules
    Args:
        args ([type]): [description]
    Returns:
        writer: tensorboard SummaryWriter
        logger: open(log.txt)
    """
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else: 
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)
    logger.debug("Debug mode activated")
    logger.info("Initialising folders ...") 
    logger.info(pprint_args(args))
    writer = SummaryWriter(args.log_dir)
    code_path = os.path.join(args.log_dir,'code_copy')
    os.makedirs(code_path, exist_ok=True)
    print('copy all file in code folder: cp -r *py %s'%code_path)
    os.system('cp -r src %s'%code_path)
    os.system('cp -r configs %s'%code_path)
    os.system('cp -r utils %s'%code_path)
    os.system('cp -r models %s'%code_path)
    return writer, logger

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

    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if logger is None:
            print('\t'.join(entries))
        else:
            logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'