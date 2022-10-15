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
      (16, 512, 8192, 64, 32768, 50264)  |    134.8    |          129.5
      (4, 2048, 8192, 64, 32768, 50264)  |    198.2    |          195.4

Times are in milliseconds (ms).

              Shape                 HF (Attn)    xFormers Cutlass (FA)
---------------------------------  -----------  -----------------------
 (8, 512, 1024, 16, 4096, 30522)     316.027            124.27
(16, 512, 8192, 64, 32768, 50264)    3462.08            1672.07
(4, 2048, 8192, 64, 32768, 50264)    9350.08            3208.07

Memory is in MBs.

[2/3] Benchmarking layer_norm
Correctness checking for Apex (FP16) is passed
Correctness checking for Triton (FP16) is passed
Correctness checking for xFormers (FP16) is passed
[---------------------------------------------------------- LayerNorm ----------------------------------------------------------]
                       |  PyTorch (FP32)  |  Apex (FP32)  |  PyTorch (FP16)  |  Apex (FP16)  |  Triton (FP16)  |  xFormers (FP16)
1 threads: ----------------------------------------------------------------------------------------------------------------------
      (32, 128, 768)   |       203.7      |      213.7    |       147.0      |      216.3    |       749.1     |        866.7
      (8, 512, 1024)   |       250.8      |      232.7    |       146.6      |      213.8    |       751.3     |        871.7
      (16, 512, 8192)  |      4523.8      |     4287.9    |      2340.0      |     2470.8    |      2102.1     |       1424.1
      (4, 2048, 8192)  |      4453.7      |     4274.0    |      2296.6      |     2527.1    |      2088.3     |       1420.3

Times are in microseconds (us).

     Shape        PyTorch (FP32)    Apex (FP32)    PyTorch (FP16)    Apex (FP16)    Triton (FP16)    xFormers (FP16)
---------------  ----------------  -------------  ----------------  -------------  ---------------  -----------------
(32, 128, 768)       60.0547          60.1484         30.0488          30.1426         30.0488           30.8008
(8, 512, 1024)       80.0586          80.1836         40.0508          40.1758         40.0508           41.0527
(16, 512, 8192)       1280.2          1281.2          640.137          641.137         640.137           642.137
(4, 2048, 8192)       1280.2          1281.2          640.137          641.137         640.137           642.137

Memory is in MBs.

[3/3] Benchmarking softmax
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/workspace_hf/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/workspace_hf/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/workspace_hf/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/workspace_hf/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module fused_mix_prec_layer_norm_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module fused_mix_prec_layer_norm_cuda...
Correctness checking for Megatron-LM (Comp-FP32) (backward) is failed: Tensor-likes are not close!

Mismatched elements: 2044451 / 16777216 (12.2%)
Greatest absolute difference: 0.047679901123046875 at index (2, 14, 373, 375) (up to 0.001 allowed)
Greatest relative difference: 1.0947146866230122 at index (2, 14, 373, 432) (up to 0.001 allowed)
Correctness checking for Megatron-LM (Comp-FP32) is passed
Correctness checking for xFormers (Comp-FP32) is passed
[------------------------------------ Softmax with FP16 input -------------------------------------]
                         |  PyTorch (Comp-FP32)  |  Megatron-LM (Comp-FP32)  |  xFormers (Comp-FP32)
1 threads: -----------------------------------------------------------------------------------------
      (4, 16, 512, 512)  |         1179.0        |           293.6           |         447.4
      (8, 16, 512, 512)  |         2325.3        |           579.3           |         446.2

Times are in microseconds (us).

      Shape         PyTorch (Comp-FP32)    Megatron-LM (Comp-FP32)    xFormers (Comp-FP32)
-----------------  ---------------------  -------------------------  ----------------------
(4, 16, 512, 512)           944                      720                      656
(8, 16, 512, 512)          1296                      848                      848

Memory is in MBs.
```

## Add an Operator

WIP
