from dataclasses import dataclass
from typing import Callable

import torch
import torch.utils.benchmark as benchmark

GLOBAL_MSGS = set()


@dataclass
class BenchConfig:
    init_func: Callable
    dtype: torch.dtype = torch.float32
    desc: str = "N/A"
    backward: bool = True
    gen_inputs: Callable = lambda shape, dtype: [torch.randn(*shape, dtype=dtype, device="cuda")]
    zero_grad: Callable = lambda func, inputs: None

    def __repr__(self):
        return f"BenchConfig({self.desc}, {self.dtype})"


@dataclass
class MemoryMeasurement:
    desc: str
    shape: str
    memory: float

    @staticmethod
    def print(results):
        from tabulate import tabulate

        descs = {} # Use dict to dedup while preserving order.
        for result in results:
            if result.desc not in descs:
                descs[result.desc] = 1
        headers = ["Shape"] + list(descs.keys())
        dict_data = {}
        for result in results:
            if result.shape not in dict_data:
                dict_data[result.shape] = {}
            dict_data[result.shape][result.desc] = result.memory

        data = []
        for shape, row in dict_data.items():
            data.append([shape] + [row[desc] for desc in headers[1:]])

        print(tabulate(data, headers=headers, stralign="center", numalign="center"))
        print("\nMemory is in MBs.\n")


def gen_output_like(func, inputs):
    with torch.no_grad():
        try:
            out = func(*inputs)
        except:
            return None
    if isinstance(out, (list, tuple)):
        ret = [torch.rand_like(o) for o in out]
    else:
        ret = torch.rand_like(out)
    return ret


def _forward_only(func, inputs, grad=None, zero_grad_fn=None):
    return func(*inputs)


def _forward_backward(func, inputs, grad, zero_grad_fn=None):
    out = func(*inputs)
    torch.autograd.backward(out, grad)
    if zero_grad_fn is not None:
        zero_grad_fn(func, inputs)
    return out


def test_func(func, inputs, grad, zero_grad_fn, verbose=False):
    try:
        if grad is not None:
            _forward_backward(func, inputs, grad, zero_grad_fn)
        else:
            _forward_only(func, inputs)
        return True
    except Exception as err:
        if verbose:
            print(err)
        return False


def skip_if(cond, desc):
    if cond:
        msg = f"Skip {desc} due to forward failure"
        if msg not in GLOBAL_MSGS:
            print(msg, flush=True)
            GLOBAL_MSGS.add(msg)
        return True
    return False


def bench(shapes, configs, label, verbose=False):
    perf_results = []
    memory_results = []
    for shape in shapes:
        for config in configs:
            func = config.init_func(shape, config.dtype)
            if skip_if(func is None, f"{config.desc}"):
                continue
            inputs = config.gen_inputs(shape, config.dtype)

            if skip_if(
                not test_func(func, inputs, None, config.zero_grad, verbose=verbose),
                f"correctness checking of {config.desc}",
            ):
                continue

            global_dict = {
                "_run": _forward_only,
                "func": func,
                "inputs": inputs,
                "zero_grad_fn": config.zero_grad,
                "grad": None,
            }
            if config.backward:
                for inp in inputs:
                    if inp is not None:
                        inp.requires_grad = inp.dtype in (torch.float32, torch.float16)
                if hasattr(func, "train"):
                    func.train()
                global_dict["_run"] = _forward_backward
                global_dict["grad"] = gen_output_like(func, inputs)

                if skip_if(
                    not test_func(
                        func, inputs, global_dict["grad"], config.zero_grad, verbose=verbose
                    ),
                    f"correctness checking of {config.desc}",
                ):
                    continue

            bencher = benchmark.Timer(
                stmt="_run(func, inputs, grad, zero_grad_fn)",
                globals=global_dict,
                label=label,
                sub_label=str(shape),
                description=config.desc,
            )
            # Benchmark. Note that this implies 500/100=5 warmups.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            perf_results.append(bencher.timeit(500))
            memory = torch.cuda.max_memory_allocated() / 2**20
            memory_results.append(MemoryMeasurement(config.desc, str(shape), memory))

            del bencher
            del global_dict
            del inputs
            torch.cuda.empty_cache()

    compare = benchmark.Compare(perf_results)
    compare.print()
    MemoryMeasurement.print(memory_results)


def check_correctness(shape, func_ref, func, config, tol=1e-5, desc="", verbose=False):
    if func is None or func_ref is None:
        print(f"Skip correctness for {desc} due to initialization failure")
        return

    inputs = config.gen_inputs(shape, config.dtype)
    if skip_if(not test_func(func, inputs, None, config.zero_grad, verbose=verbose), f"{desc}"):
        return

    if config.backward:
        for inp in inputs:
            if inp is not None:
                inp.requires_grad = inp.dtype in (torch.float32, torch.float16)
        grads_input = gen_output_like(func_ref, inputs)
        out_ref = _forward_backward(func_ref, inputs, grads_input)
        grads_ref = [inp.grad for inp in inputs if inp is not None]
        config.zero_grad(func_ref, inputs)

        if skip_if(
            not test_func(func, inputs, grads_input, config.zero_grad, verbose=verbose),
            f"{desc}",
        ):
            return
        out = _forward_backward(func, inputs, grads_input)
        grads = [inp.grad for inp in inputs if inp is not None]
        config.zero_grad(func, inputs)
    else:
        out_ref = _forward_only(func_ref, inputs)
        out = _forward_only(func, inputs)

    # Check forward.
    try:
        torch.testing.assert_close(out, out_ref, rtol=tol, atol=tol)
    except Exception as err:
        print(f"Correctness checking for {desc} (forward) is failed: {err}", flush=True)

    # Check backward.
    if config.backward:
        for grad, grad_ref in zip(grads, grads_ref):
            try:
                torch.testing.assert_close(grad, grad_ref, rtol=tol, atol=tol)
            except Exception as err:
                print(f"Correctness checking for {desc} (backward) is failed: {err}", flush=True)

    print(f"Correctness checking for {desc} is passed", flush=True)
