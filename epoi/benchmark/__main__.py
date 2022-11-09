"""Entry point."""
import argparse
import sys
import traceback
from inspect import getmembers, getmodule, isfunction, ismodule

from collections import OrderedDict

from . import fused_ops, layer_ops, norm_ops
from . import logger
from .utils import get_version_n_commit

logger = logger.get_logger("main")

LIBS = ["epoi", "transformers", "xformers", "megatron", "triton", "apex"]


def get_case_list():
    self = sys.modules[__name__]
    funcs = []
    for name, mod in getmembers(self):
        if not ismodule(mod) or not name.endswith("ops"):
            continue
        for name, func in getmembers(mod):
            if isfunction(func) and getmodule(func) == mod:
                funcs.append((name, func))
    return funcs


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Efficient PyTorch Ops")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward ops")
    parser.add_argument(
        "--only-run",
        type=str,
        help="Only run the ops that contain this string."
        "You can use ',' to separate multiple strings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print error messages when failures",
    )
    return parser.parse_args()


def select(only_run, name):
    if only_run is None:
        return True
    return any([s in name for s in only_run])


def list_envs():
    from tabulate import tabulate
    import torch

    print("===== Environment =====\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # PyTorch
    print("PyTorch Configuration")
    data = OrderedDict()
    ver, commit = get_version_n_commit("torch")
    data["Version"] = ver
    if commit != "N/A":
        data["Commit"] = commit
    data["Built w. CUDA"] = torch.version.cuda
    print(tabulate(data.items(), headers=["Config", "Value"], stralign="center", numalign="center"))
    print("\n")

    # Other libs
    print("Other Libraries Configuration")
    data = [[lib] + list(get_version_n_commit(lib)) for lib in LIBS]
    data = [info for info in data if info[1] != "N/A" or info[2] != "N/A"]
    print(
        tabulate(
            data, headers=["Package", "Version", "Commit SHA"], stralign="center", numalign="center"
        )
    )
    print("===== Environment =====\n")


def main():
    args = parse_args()
    only_run = None if args.only_run is None else args.only_run.split(",")

    list_envs()

    funcs = []
    selected, total = 0, 0
    for name, func in get_case_list():
        if select(only_run, name):
            logger.info(f"Selected {name}")
            funcs.append((name, func))
            selected += 1
        else:
            logger.info(f"Skipped {name}")
        total += 1
    logger.info(f"Running selected {selected}/{total} cases")

    n_func = len(funcs)
    for idx, (name, func) in enumerate(funcs):
        logger.info(f"[{idx + 1}/{n_func}] Benchmarking {name}")
        try:
            func(args)
        except Exception as err:
            traceback.print_exc()
            logger.info(f"Failed to benchmark {name}: {err}")


if __name__ == "__main__":
    main()
