# Efficient PyTorch Operator Inventory (EPOI)

This inventory includes efficient PyTorch custom operators for training.
It also inclues a benchmark suite to easily evaluate their latencies and memory usages.

## Requirements

The covered operators may have dependencies to other libraries listed as follows.
It is recommended to intall all of them to obtain a complete benchmark. However
it's fine if you just want to benchmark the operators from certain libraries.

- HuggingFace transformers (https://github.com/huggingface/transformers): Any installation works.
- NVIDIA Apex (https://github.com/NVIDIA/apex): Clone and use setup.py to build from source.
- Megatron-LM (https://github.com/NVIDIA/Megatron-LM): Clone and add the path to PYTHONPATH.
- xFormers (https://github.com/facebookresearch/xformers): Clone and use setup.py to build from source. Verified commit: 48a77cc

## Inventory

You can easily use the covered operators in your PyTorch models:

```python
import torch
from epoi.ops.xformers_attn import BertSelfAttention as EpoiBertSelfAttention

class Model(torch.nn.Module):
      def __init__(self):
            super().__init__()
            self.attn = EpoiBertSelfAttention(...)

      def forward(self, hidden_states):
            out = self.attn(hidden_states)
            ...
```

## Benchmarking

Note that you need to install the corresponding packages (e.g., apex)
to import/benchmark certain operators.

```
python -m epoi.benchmark
```

This will benchmark all included operators on your local GPUs. The full benchmark
results can be found [here](notebooks/benchmark-ops.ipynb).


In addition, the following flags may also useful:

`--only-run op1,op2`: Only benchmark the ops with `op1` OR `op2` in their names.
You can use comma to specify more ops at once.

`--forward-only`: The deafult benchmark includes a forward and a backward. If you only want
to benchmark the forward part, specify this flag in your command.

`--verbose`: You may find some ops failed to be benchmarked. Possible reasons include
out of memory or missing dependencies (e.g., apex, triton, xformers, etc).
In this case, you can use this flag to see a complete error message for debugging.

Example (on NVIDIA V100):

```
python -m epoi.benchmark --only-run gpt_attention
```

```
===== Environment =====

GPU: Tesla V100-SXM2-16GB

PyTorch Configuration
   Config         Value
-------------  ------------
   Version     1.12.1+cu116
Built w. CUDA      11.6


Other Libraries Configuration
  Package       Version                   Commit SHA
------------  -----------  ----------------------------------------
    epoi        0.1.dev    094608d0759392516d5c6b4e00e00e72b3156c1c
transformers  4.24.0.dev0  12ce2941c7b67c0dedac0f0468b3ed854fa940ab
  xformers    0.0.14.dev   ba93c5012d00bd1b010514a7bc9bd938c1ad6149
   triton        2.0.0                       N/A
    apex          0.1                        N/A
===== Environment =====

[2022-10-28 00:35:18] INFO main: Skipped bias_gelu
[2022-10-28 00:35:18] INFO main: Skipped dropout_add_ln
[2022-10-28 00:35:18] INFO main: Skipped bert_attention
[2022-10-28 00:35:18] INFO main: Selected gpt_attention
[2022-10-28 00:35:18] INFO main: Skipped qkv_self_attn
[2022-10-28 00:35:18] INFO main: Skipped layer_norm
[2022-10-28 00:35:18] INFO main: Skipped softmax
[2022-10-28 00:35:18] INFO main: Running selected 1/7 cases
[2022-10-28 00:35:18] INFO main: [1/1] Benchmarking gpt_attention
[2022-10-28 00:35:23] INFO bencher: Correctness checking for xFormers FlashAttn (cutlass) is passed
[2022-10-28 00:35:23] WARNING bencher: Skip correctness checking for xFormers FlashAttn (triton): Forward failed
[----- GPT Attention (Attn) and FlashAttention (FA) without mask ------]
                                  |  HF (Attn)  |  xFormers cutlass (FA)
1 threads: -------------------------------------------------------------
      (8, 1024, 1024, 16, 50257)  |     14.9    |            6.0
      (16, 512, 8192, 64, 50264)  |    184.4    |          164.8
      (4, 2048, 8192, 64, 50264)  |    261.3    |          197.9

Times are in milliseconds (ms).

          Shape              HF (Attn)    xFormers cutlass (FA)
--------------------------  -----------  -----------------------
(8, 1024, 1024, 16, 50257)     1091              178.502
(16, 512, 8192, 64, 50264)    2688.27            1284.02
(4, 2048, 8192, 64, 50264)    8836.02            1284.02

Memory is in MBs and excludes inputs/outputs.
```

## Module Injection

EPOI also provides two approaches for you to inject the covered moduels to your model
as long as your model has a corresponding policy that specifies how to inject modules.

If your model doesn't have a builtin injection policy, you could also custom one and
register it:

```python
from epoi.inject import register_policy, ModuleInjectPolicy

@register_policy
class MyPolicy(ModuleInjectPolicy):
  # Implement you policy (tutorial TBA)
```

### Module Injection after Initialization

If you prefer to initialize the model first, you could inject modules as follows:

```python
from epoi.inject import inject_module

model = init_model()
inject_module(model)
```

You can refer to [this Jupyter notebook](notebooks/hf-clm-with-injection-single-device.ipynb) that uses this approach to inject modules to GPT2-medium.

### Module Injection during Initialization

**Note that this approach doesn't support model loading from a checkpoint yet.**

If you have to inject modules during model initialization (e.g., train the model with ZeRO-3),
you could inject modules as follows:

```python
from epoi.inject import InjectModuleContext

with InjectModuleContext():
  ...
  model = init_model()
```

You can refer to [this Jupyter notebook](notebooks/hf-clm-with-injection-deepspeed.ipynb) that uses this approach
to inject modules to GPT2-xl and trains with DeepSpeed ZeRO-3.
