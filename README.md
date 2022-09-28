# Benchmark PyTorch Operators

This benchmark suite includes popular PyTorch custom operators
for transformer model training.

## Usage

Note that you need to install the corresponding packages (e.g., apex)
to benchmark certain operators.

```
python -m bench_pt_op
```

Example results (on NVIDIA V100):

```
[1 / 2] Benchmarking dropout_add_ln
[--------------------------------------- Dropout+LayerNorm ---------------------------------------]
                       |  Eager (FP32)  |  TS+nvFuser (FP32)  |  Eager (FP16)  |  TS+nvFuser (FP16)
1 threads: ----------------------------------------------------------------------------------------
      (32, 128, 768)   |     306.2      |        255.0        |     235.4      |        268.6
      (4, 512, 768)    |     234.3      |        257.8        |     234.7      |        269.5
      (16, 512, 768)   |     579.2      |        380.5        |     325.9      |        267.4
      (64, 128, 1024)  |     772.5      |        505.1        |     411.3      |        297.2

Times are in microseconds (us).

[2 / 2] Benchmarking layer_norm
[-------------------------------------- LayerNorm ---------------------------------------]
                      |  PyTorch (FP32)  |  Apex (FP32)  |  PyTorch (FP16)  |  Apex (FP16)
1 threads: -------------------------------------------------------------------------------
      (32, 128, 768)  |      192.0       |     213.7     |      139.8       |     213.6
      (16, 512, 768)  |      363.4       |     333.4     |      204.2       |     211.3

Times are in microseconds (us).
```

## Add an Operator

WIP
