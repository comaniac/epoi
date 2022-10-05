from dataclasses import dataclass
from typing import Callable

import torch
import torch.utils.benchmark as benchmark


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
        return torch.rand_like(out)

    def _forward_only(func, inputs, grad, zero_grad_fn):
        return func(*inputs)

    def _forward_backward(func, inputs, grad, zero_grad_fn):
        out = func(*inputs)
        out.backward(grad)
        zero_grad_fn(func, inputs)
        return out

    results = []
    for shape in shapes:
        for config in configs:
            func = config.init_func(shape, config.dtype)
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
            # Benchmark. Note that this implies 1000/100=10 warmups.
            results.append(bencher.timeit(1000))

    compare = benchmark.Compare(results)
    compare.print()
