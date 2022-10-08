"""Ops for benchmarking.

For shapes to benchmark, here are common configurations for transformaer models:

                   batch,  seq,  head, hidden, intermediate, vocab
bert-large      :   32,    512,  16,   768,    1024,         30522
gigantic model 1:   16,    512,  64,   8192,   32768,        32008 or 50264 or 256032
gigantic model 2:   4,     2048, 64,   8192,   32768,        32008 or 50264 or 256032
"""
from inspect import getmembers, getmodule, isfunction, ismodule
import sys

from . import apex_ops
from . import torchscript_ops
from . import struct_opt_ops
from . import megatron_ops

def get_op_list():
    self = sys.modules[__name__]
    funcs = []
    for name, mod in getmembers(self):
        if not ismodule(mod) or not name.endswith("ops"):
            continue
        for name, func in getmembers(mod):
            if isfunction(func) and getmodule(func) == mod:
                funcs.append((name, func))
    return funcs
