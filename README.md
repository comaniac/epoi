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
- xFormers (https://github.com/facebookresearch/xformers): Clone and use setup.py to build from source.

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

This will benchmark all included operators on your local GPUs.
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
python -m epoi.benchmark --only-run bert_attention,softmax,layer_norm
```

```
Skipped dropout_add_ln
Skipped megatron_bias_gelu
Selected bert_attention
Skipped gpt_attention
Skipped qkv_self_attn
Selected layer_norm
Selected softmax
Running selected 3/7 cases
[1/3] Benchmarking bert_attention
Skip xFormers FlashAttn due to forward failure
Correctness checking for xFormers Cutlass FlashAttn is passed
Skip xFormers Triton FlashAttn due to forward failure
Skip correctness checking of xFormers (FA) due to forward failure
Skip correctness checking of xFormers Triton (FA) due to forward failure
[------------------ Attention (Attn) and FlashAttention (FA) -----------------]
                                         |  HF (Attn)  |  xFormers Cutlass (FA)
1 threads: --------------------------------------------------------------------
      (8, 512, 1024, 16, 4096, 30522)    |      4.1    |            2.7
      (16, 512, 8192, 64, 32768, 50264)  |    135.3    |          129.5
      (4, 2048, 8192, 64, 32768, 50264)  |    198.2    |          195.4

Times are in milliseconds (ms).

              Shape                 xFormers Cutlass (FA)    HF (Attn)
---------------------------------  -----------------------  -----------
 (8, 512, 1024, 16, 4096, 30522)           124.27             316.027
(16, 512, 8192, 64, 32768, 50264)          1672.07            3462.08
(4, 2048, 8192, 64, 32768, 50264)          3208.07            9350.08

Memory is in MBs.

[2/3] Benchmarking layer_norm
Correctness checking for Apex (FP16) is passed
Correctness checking for Triton (FP16) is passed
Correctness checking for xFormers (FP16) is passed
[---------------------------------------------------------- LayerNorm ----------------------------------------------------------]
                       |  PyTorch (FP32)  |  Apex (FP32)  |  PyTorch (FP16)  |  Apex (FP16)  |  Triton (FP16)  |  xFormers (FP16)
1 threads: ----------------------------------------------------------------------------------------------------------------------
      (32, 128, 768)   |       194.6      |      205.6    |       138.6      |      204.0    |       707.4     |        844.9
      (8, 512, 1024)   |       251.0      |      232.6    |       136.9      |      202.8    |       719.4     |        821.1
      (16, 512, 8192)  |      4574.2      |     4297.5    |      2309.2      |     2482.4    |      2102.0     |       1420.8
      (4, 2048, 8192)  |      4465.6      |     4300.8    |      2307.9      |     2501.9    |      2095.6     |       1419.5

Times are in microseconds (us).

     Shape        PyTorch (FP16)    PyTorch (FP32)    xFormers (FP16)    Triton (FP16)    Apex (FP16)    Apex (FP32)
---------------  ----------------  ----------------  -----------------  ---------------  -------------  -------------
(32, 128, 768)       30.0488           60.0547            30.8008           30.0488         30.1426        60.1484
(8, 512, 1024)       40.0508           80.0586            41.0527           40.0508         40.1758        80.1836
(16, 512, 8192)      640.137            1280.2            642.137           640.137         641.137        1281.2
(4, 2048, 8192)      640.137            1280.2            642.137           640.137         641.137        1281.2

Memory is in MBs.

[3/3] Benchmarking softmax
Correctness checking for Megatron-LM (Comp-FP32) (backward) is failed: Tensor-likes are not close!

Mismatched elements: 2047251 / 16777216 (12.2%)
Greatest absolute difference: 0.0552825927734375 at index (2, 13, 150, 350) (up to 0.001 allowed)
Greatest relative difference: 1.1963111760409058 at index (2, 13, 150, 224) (up to 0.001 allowed)
Correctness checking for Megatron-LM (Comp-FP32) is passed
Correctness checking for xFormers (Comp-FP32) is passed
[------------------------------------ Softmax with FP16 input -------------------------------------]
                         |  PyTorch (Comp-FP32)  |  Megatron-LM (Comp-FP32)  |  xFormers (Comp-FP32)
1 threads: -----------------------------------------------------------------------------------------
      (4, 16, 512, 512)  |         1179.3        |           293.9           |         418.4
      (8, 16, 512, 512)  |         2325.7        |           579.7           |         434.0

Times are in microseconds (us).

      Shape         xFormers (Comp-FP32)    Megatron-LM (Comp-FP32)    PyTorch (Comp-FP32)
-----------------  ----------------------  -------------------------  ---------------------
(4, 16, 512, 512)           656                       720                      944
(8, 16, 512, 512)           384                       848                     1296

Memory is in MBs.

```

## Add an Operator

WIP
