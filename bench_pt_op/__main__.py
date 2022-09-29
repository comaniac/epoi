"""Entry point."""
from inspect import getmembers, getmodule, isfunction

from . import ops


def main():
    funcs = []
    for name, func in getmembers(ops):
        if isfunction(func) and getmodule(func) == ops:
            funcs.append((name, func))

    n_func = len(funcs)
    for idx, (name, func) in enumerate(funcs):
        print(f"[{idx + 1} / {n_func}] Benchmarking {name}", flush=True)
        try:
            func()
        except Exception as err:
            print(f"Failed: {str(err)}")


if __name__ == "__main__":
    main()
