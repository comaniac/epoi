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


def bench(shapes, configs, label):
    def gen_output_like(func, inputs):
        with torch.no_grad():
            out = func(*inputs)
        if isinstance(out, (list, tuple)):
            ret = [torch.rand_like(o) for o in out]
        else:
            ret = torch.rand_like(out)
        return ret

    def _forward_only(func, inputs, grad, zero_grad_fn):
        return func(*inputs)

    def _forward_backward(func, inputs, grad, zero_grad_fn):
        out = func(*inputs)
        torch.autograd.backward(out, grad)
        zero_grad_fn(func, inputs)
        return out

    results = []
    for shape in shapes:
        for config in configs:
            func = config.init_func(shape, config.dtype)
            if func is None:  # Skip if failed to initialize.
                msg = f"Skip {config.desc} due to initialization failure"
                if msg not in GLOBAL_MSGS:
                    print(msg)
                    GLOBAL_MSGS.add(msg)
                continue
            inputs = config.gen_inputs(shape, config.dtype)

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
                global_dict["_run"] = _forward_backward
                global_dict["grad"] = gen_output_like(func, inputs)

            bencher = benchmark.Timer(
                stmt="_run(func, inputs, grad, zero_grad_fn)",
                globals=global_dict,
                label=label,
                sub_label=str(shape),
                description=config.desc,
            )
            # Benchmark. Note that this implies 500/100=5 warmups.
            results.append(bencher.timeit(500))

            del bencher
            del global_dict
            del inputs
            torch.cuda.empty_cache()

    compare = benchmark.Compare(results)
    compare.print()
