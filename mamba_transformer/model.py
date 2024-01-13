import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock, FeedForward, MultiQueryAttention
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor):
        return F.normalize(x, dim = - 1) * self.scale * self.g


class MultiQueryTransformerBlock(nn.Module):
    """
    MultiQueryTransformerBlock is a module that represents a single block of the Multi-Query Transformer.
    It consists of a multi-query attention layer, a feed-forward network, and layer normalization.

    Args:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.

    Attributes:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout probability.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        attn (MultiQueryAttention): The multi-query attention layer.
        ffn (FeedForward): The feed-forward network.
        norm (nn.LayerNorm): The layer normalization.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs a forward pass of the MultiQueryTransformerBlock.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult

        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the MultiQueryTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x, _, _ = self.attn(x)
        x = self.norm(x)
        x = self.ffn(x)
        return x


class MambaTransformerblock(nn.Module):
    """
    MambaTransformerblock is a module that represents a block in the Mamba Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads in the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
        d_state (int, optional): The dimension of the state. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        d_state (int): The dimension of the state.
        dropout (float): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        mamba_blocks (nn.ModuleList): List of MambaBlock instances.
        transformer_blocks (nn.ModuleList): List of MultiQueryTransformerBlock instances.
        ffn_blocks (nn.ModuleList): List of FeedForward instances.
        norm (nn.LayerNorm): Layer normalization module.
        
    Examples:
        import torch 
        from mt import MambaTransformerblock
        
        x = torch.randn(1, 10, 512)
        model = MambaTransformerblock(
            dim=512,
            heads=8,
            depth=4,
            dim_head=64,
            d_state=512,
            dropout=0.1,
            ff_mult=4
        )
        print(model(x).shape)


    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state

        self.mamba_blocks = nn.ModuleList([])
        self.transformer_blocks = nn.ModuleList([])
        self.ffn_blocks = nn.ModuleList([])

        self.mamba_blocks.append(
            MambaBlock(dim, depth, d_state, *args, **kwargs)
        )

        # Transformer and ffn blocks
        for _ in range(depth):
            self.transformer_blocks.append(
                MultiQueryTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    dropout,
                    ff_mult,
                    *args,
                    **kwargs,
                )
            )

            self.ffn_blocks.append(
                FeedForward(dim, dim, ff_mult, *args, **kwargs)
            )

        # Layernorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for mamba, attn, ffn in zip(
            self.mamba_blocks,
            self.transformer_blocks,
            self.ffn_blocks,
        ):
            x = self.norm(x)
            x = mamba(x) + x
            x = self.norm(x)
            x = attn(x) + x
            x = self.norm(x)
            x = ffn(x) + x

        return x


class MambaTransformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        *args,
        **kwargs,
    ):
        """
        MambaTransformer is a PyTorch module that implements the Mamba Transformer model.

        Args:
            num_tokens (int): The number of tokens in the input vocabulary.
            dim (int): The dimensionality of the token embeddings and model hidden states.
            heads (int): The number of attention heads.
            depth (int): The number of transformer blocks.
            dim_head (int): The dimensionality of each attention head.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
            d_state (int, optional): The dimensionality of the state embeddings. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state
        
        self.emb = nn.Embedding(num_tokens, dim)
        self.mt_block = MambaTransformerblock(
            dim,
            heads,
            depth,
            dim_head,
            dropout,
            ff_mult,
            d_state,
            *args,
            **kwargs,
        )
        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaTransformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, num_tokens).
        """
        x = self.emb(x)
        x = self.mt_block(x)
        return self.to_logits(x)
    
    
