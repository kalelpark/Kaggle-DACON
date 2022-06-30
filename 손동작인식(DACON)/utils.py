import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

tqdm_kwargs = dict(ncols=128, file=sys.stdout, leave=False)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


class AverageMeter(object):
    """
    AverageMeter, referenced to https://dacon.io/competitions/official/235626/codeshare/1684
    """

    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class CustomLogger:
    def __init__(self, filename=None, filemode="a", use_color=True):
        if filename is not None:
            self.empty = False
            filename = Path(filename)
            if filename.is_dir():
                timestr = self._get_timestr().replace(" ", "_").replace(":", "-")
                filename = filename / f"log_{timestr}.log"
            self.file = open(filename, filemode)
        else:
            self.empty = True

        self.use_color = use_color

    def _get_timestr(self):
        n = datetime.now()
        return f"{n.year:04d}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"

    def _write(self, msg, level):
        timestr = self._get_timestr()
        out = f"[{timestr} {level}] {msg}"

        if self.use_color:
            if level == " INFO":
                print("\033[34m" + out + "\033[0m")
            elif level == " WARN":
                print("\033[35m" + out + "\033[0m")
            elif level == "ERROR":
                print("\033[31m" + out + "\033[0m")
            elif level == "FATAL":
                print("\033[43m\033[1m" + out + "\033[0m")
            else:
                print(out)
        else:
            print(out)

        if not self.empty:
            self.file.write(out + "\r\n")

    def debug(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "DEBUG")

    def info(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " INFO")

    def warn(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " WARN")

    def error(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "ERROR")

    def fatal(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "FATAL")

    def flush(self):
        if not self.empty:
            self.file.flush()


def timenow(braket=False):
    n = datetime.now()
    if braket:
        return f"[{n.year}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}]"
    else:
        return f"{n.year}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"


def dict_cuda(d, device=torch.device("cuda")):
    o = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            o[k] = v.to(device, non_blocking=True)
        else:
            o[k] = v
    return o


def make_result_dir(config):
    root_dir = Path(config.result_dir_root)
    pp = root_dir / f"exp{config.exp_num}"
    for i in range(999, -1, -1):
        ppf = pp / f"version_{i:03d}"
        if ppf.exists():
            break

    ppf = pp / f"version_{i + 1:03d}"
    ppf.mkdir(parents=True, exist_ok=True)
    config.ver_num = i + 1

    return ppf


def strize_dict(dic: dict):
    new_dic = OrderedDict()
    for k, v in dic.items():
        new_dic[k] = str(v)
        # if isinstance(v, dict):
        #     new_dic[k] = strize_dict(v)
        # else:
        #     new_dic[k] = str(v)
    return new_dic


class ChainDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.total_len = reduce(lambda a, b: a + len(b), self.datasets, 0)

        self.acc_lens = [None for _ in range(self.total_len)]
        i = 0
        for k, ds in enumerate(self.datasets):
            for j in range(i, i + len(ds)):
                self.acc_lens[j] = (k, i)
            i += len(ds)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx_):
        k, j = self.acc_lens[idx_]
        ds = self.datasets[k]
        idx = idx_ - j
        return ds[idx]
