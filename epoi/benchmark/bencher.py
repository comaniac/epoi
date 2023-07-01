from dataclasses import dataclass
from typing import Callable, Tuple
import gc
import traceback

import torch
import torch.utils.benchmark as benchmark

from .logger import get_logger

logger = get_logger("bencher")

GLOBAL_MSGS = set()


@dataclass
class BenchConfig:
    init_func: Callable
    dtype: torch.dtype = torch.float32
    desc: str = "N/A"
    backward: bool = True
    requires_grad: Tuple[bool] = (True, False, ...)
    inputs_requires_grad: Tuple[bool] = (True, ...)
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
    def print(results, note=None):
        from tabulate import tabulate

        descs = {}  # Use dict to dedup while preserving order.
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
            data.append([shape] + [row[desc] if desc in row else "N/A" for desc in headers[1:]])

        print(tabulate(data, headers=headers, stralign="center", numalign="center"))
        print("\nMemory is in MBs and excludes inputs/outputs.")
        if note:
            print(note)
        print()


def expand_requires_grad(requires_grad, target_len):
    requires_grad = list(requires_grad[:-1]) + [requires_grad[-2]] * (
        target_len - len(requires_grad) + 1
    )
    return requires_grad


def gen_grad(func, inputs, requires_grad):
    with torch.no_grad():
        try:
            out = func(*inputs)
        except:
            return None

    if isinstance(out, (list, tuple)):
        if len(out) > len(requires_grad) - 1:
            if not isinstance(requires_grad[-1], type(...)):
                raise ValueError(
                    "requires_grad must have the same length as out, "
                    "or the end with '...' to repeat the last value."
                )
            else:
                requires_grad = expand_requires_grad(requires_grad, len(out))
        ret = [torch.ones_like(o) if r else None for o, r in zip(out, requires_grad)]
    else:
        assert requires_grad[0], "Single output must require grad"
        ret = torch.ones_like(out)
    return ret


def set_inputs_requires_grad(inputs, inputs_requires_grad):
    if isinstance(inputs, (list, tuple)):
        if len(inputs) > len(inputs_requires_grad) - 1:
            if not isinstance(inputs_requires_grad[-1], type(...)):
                raise ValueError(
                    "inputs_requires_grad must have the same length as inputs, "
                    "or the end with '...' to repeat the last value."
                )
            else:
                inputs_requires_grad = expand_requires_grad(inputs_requires_grad, len(inputs))
        for inp, requires_grad in zip(inputs, inputs_requires_grad):
            if inp is not None:
                inp.requires_grad = requires_grad and (inp.dtype in (torch.float32, torch.float16))
    else:
        assert inputs_requires_grad[0], "Single input must require grad"
        inputs.requires_grad = True



def _forward_only(func, inputs, grad=None, zero_grad_fn=None):
    ret = func(*inputs)
    torch.cuda.synchronize()
    return ret


def _forward_backward(func, inputs, grad, zero_grad_fn=None):
    out = func(*inputs)
    if isinstance(out, (list, tuple)):
        assert len(out) == len(grad), f"len(out)={len(out)} != len(grad)={len(grad)}"
        target_out = []
        target_grad = []
        for o, g in zip(out, grad):
            if g is not None:
                target_out.append(o)
                target_grad.append(g)
    torch.autograd.backward(target_out, target_grad)
    if zero_grad_fn is not None:
        zero_grad_fn(func, inputs)
    torch.cuda.synchronize()
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
            print(traceback.format_exc())
            logger.warning(err)
        return False


def skip_if(cond, desc):
    if cond:
        msg = f"Skip {desc}"
        if msg not in GLOBAL_MSGS:
            logger.warning(msg)
            GLOBAL_MSGS.add(msg)
        return True
    return False


def print_live_tensors():
    gc.collect()
    tc = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                print("GC Tensor", type(obj), obj.size())
                tc += obj.numel()
        except:
            pass


def bench(shapes, configs, label, verbose=False):
    perf_results = []
    memory_results = []
    for shape in shapes:
        for config in configs:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            func = config.init_func(shape, config.dtype)
            if skip_if(func is None, f"{config.desc}: Initialization failed with shape {shape}"):
                continue
            inputs = config.gen_inputs(shape, config.dtype)

            if skip_if(
                not test_func(func, inputs, None, config.zero_grad, verbose=verbose),
                f"{config.desc}: Forward failed with shape {shape}",
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
                global_dict["_run"] = _forward_backward
                global_dict["grad"] = gen_grad(func, inputs, config.requires_grad)
                set_inputs_requires_grad(inputs, config.inputs_requires_grad)
                if hasattr(func, "train"):
                    func.train()

                if skip_if(
                    not test_func(
                        func, inputs, global_dict["grad"], config.zero_grad, verbose=verbose
                    ),
                    f"{config.desc}: Backward failed with shape {shape}",
                ):
                    continue

            bencher = benchmark.Timer(
                stmt="_run(func, inputs, grad, zero_grad_fn)",
                globals=global_dict,
                label=label,
                sub_label=str(shape),
                description=config.desc,
            )
            # Benchmark memory. Note that we only run forward.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.max_memory_allocated() / 2**20
            _forward_only(func, inputs)
            memory_after = torch.cuda.max_memory_allocated() / 2**20
            memory_results.append(
                MemoryMeasurement(config.desc, str(shape), memory_after - memory_before)
            )

            # Benchmark latency. Note that this implies 500/100=5 warmups.
            perf_results.append(bencher.timeit(500))

    compare = benchmark.Compare(perf_results)
    compare.print()
    MemoryMeasurement.print(
        memory_results, "Note that memory is measured with only forward (but with activations)."
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return compare, memory_results


def check_correctness(
    shape, func_ref, func, config, fwd_tol=5e-2, bwd_tol=5e-2, desc="", verbose=False
):
    if func is None or func_ref is None:
        logger.warning(f"Correctness checking for {desc} failed at initialization")
        return None

    inputs = config.gen_inputs(shape, config.dtype)
    if skip_if(
        not test_func(func, inputs, None, config.zero_grad, verbose=verbose),
        f"correctness checking for {desc}: Forward failed",
    ):
        return None

    if config.backward:
        set_inputs_requires_grad(inputs, config.inputs_requires_grad)
        grads_input = gen_grad(func_ref, inputs, config.requires_grad)
        out_ref = _forward_backward(func_ref, inputs, grads_input)
        grads_ref = [inp.grad for inp in inputs if (inp is not None) and inp.requires_grad]
        config.zero_grad(func_ref, inputs)

        if skip_if(
            not test_func(func, inputs, grads_input, config.zero_grad, verbose=verbose),
            f"correctness checking for {desc}: Backward failed",
        ):
            return None
        out = _forward_backward(func, inputs, grads_input)
        grads = [inp.grad for inp in inputs if (inp is not None) and inp.requires_grad]
        config.zero_grad(func, inputs)
    else:
        out_ref = _forward_only(func_ref, inputs)
        out = _forward_only(func, inputs)

    # Check forward.
    try:
        torch.testing.assert_close(out, out_ref, atol=fwd_tol, rtol=fwd_tol)
    except Exception as err:
        logger.warning(f"Correctness checking for {desc} (forward) is failed: {err}")
        return False

    # Check backward.
    if config.backward:
        for grad, grad_ref in zip(grads, grads_ref):
            try:
                torch.testing.assert_close(grad, grad_ref, atol=bwd_tol, rtol=bwd_tol)
            except Exception as err:
                logger.warning(f"Correctness checking for {desc} (backward) is failed: {err}")
                return False

    logger.info(f"Correctness checking for {desc} is passed")
    return True
