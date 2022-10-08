"""Model structure optimized ops."""
import torch

from ..bench import BenchConfig, bench


def qkv_self_attn(args):
    class UnfusedModule(torch.nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.key = torch.nn.Linear(hidden_size, num_heads * self.head_size)
            self.value = torch.nn.Linear(hidden_size, num_heads * self.head_size)
            self.query = torch.nn.Linear(hidden_size, num_heads * self.head_size)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def forward(self, hidden_states):
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            return (query_layer, key_layer, value_layer)

    class FusedModule(torch.nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.qkv = torch.nn.Linear(hidden_size, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            # (B, S, 3 * H) -> (B, S, D, H/D, 3)
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size, 3)
            x = x.view(*new_x_shape)
            # (B, S, D, H/D, 3) -> (B, D, S, H/D, 3)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):
            combined_layer = self.qkv(hidden_states)
            # (B, S, 3 * H) -> (B, D, S, H/D, 3)
            combined_layer = self.transpose_for_scores(combined_layer)
            # (B, D, S, H/D, 3) -> 3 * (B, D, S, H/D)
            return [torch.squeeze(t) for t in torch.split(combined_layer, 1, dim=-1)]

    def _init(shape, dtype, fused):
        model = UnfusedModule(*shape[2:]) if not fused else FusedModule(*shape[2:])
        if dtype == torch.float16:
            model = model.half()
        return model.cuda()

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape[:-1], dtype=dtype, device="cuda")
        return [inp]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        if hasattr(mod, "qkv"):
            mod.qkv.weight.grad = None
            mod.qkv.bias.grad = None
        else:
            mod.key.weight.grad = None
            mod.key.bias.grad = None
            mod.value.weight.grad = None
            mod.value.bias.grad = None
            mod.query.weight.grad = None
            mod.query.bias.grad = None

    no_fuse = lambda shape, dtype: _init(shape, dtype, False)
    fuse = lambda shape, dtype: _init(shape, dtype, True)

    # (batch, seq, hidden size, num head)
    shapes = [
        (4, 512, 1024, 16),  # bert-large w. BS4, seq512
        (8, 512, 1024, 16),  # bert-large w. BS8, seq512
        (16, 512, 1024, 16),  # bert-large w. BS16, seq512
        (16, 512, 8192, 64),  # gigantic model 1 w. BS16, seq512
        (4, 2048, 8192, 64),  # gigantic model 1 w. BS4, seq2048
    ]
    bench(
        shapes,
        [
            BenchConfig(
                no_fuse,
                torch.float16,
                "NoFuse (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fuse,
                torch.float16,
                "Fused (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "QKV in Self-Attention",
    )
