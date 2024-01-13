from torch import nn, einsum

from einops import rearrange

from zeta.utils import exists

# linear attention


class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        if exists(mask):
            k.masked_fill_(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, " (b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
