"""Entry point."""
import traceback
import argparse

from .ops import get_op_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward ops")
    parser.add_argument(
        "--only-run",
        type=str,
        help="Only run the ops that contain this string."
        "You can use ',' to separate multiple strings",
    )
    return parser.parse_args()


def select(only_run, name):
    if only_run is None:
        return True
    return any([s in name for s in only_run])


def main():
    args = parse_args()
    only_run = None if args.only_run is None else args.only_run.split(",")

    funcs = []
    selected, total = 0, 0
    for name, func in get_op_list():
        if select(only_run, name):
            print(f"Selected {name}")
            funcs.append((name, func))
            selected += 1
        else:
            print(f"Skipped {name}")
        total += 1
    print(f"Running selected {selected}/{total} ops")

    n_func = len(funcs)
    for idx, (name, func) in enumerate(funcs):
        print(f"[{idx + 1}/{n_func}] Benchmarking {name}", flush=True)
        try:
            func(args)
        except Exception as err:
            traceback.print_exc()
            print(f"Failed to benchmark {name}: {err}")


if __name__ == "__main__":
    main()
