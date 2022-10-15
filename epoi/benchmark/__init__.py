"""Benchmarking cases.

For shapes to benchmark, here are common configurations for transformaer models:

                   batch,  seq,  head, hidden, intermediate, vocab
bert-large      :   32,    512,  16,   768,    1024,         30522
gigantic model 1:   16,    512,  64,   8192,   32768,        32008 or 50264 or 256032
gigantic model 2:   4,     2048, 64,   8192,   32768,        32008 or 50264 or 256032
"""