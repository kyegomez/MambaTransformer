[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Mamba Transformer

![Mamba Transformer](https://www.figma.com/file/YZxnPMjtj5XrEA4XC2ANIE/Mamba-Transformer?type=whiteboard&node-id=1%3A2&t=ybC6tnz8xiYie1hb-1?raw=true)

Integrating Mamba/SSMs with Transformer for Enhanced Long Context and High-Quality Sequence Modeling.

This is 100% novel architecture that I have designed to combine the strengths and weaknesses out of SSMs and Attention for an all-new advanced architecture with the purpose of surpassing our old limits. Faster processing speed, longer context lengths, lower perplexity over long sequences, enhanced and superior reasoning while remaining small and compact.

The architecture is essentially: `x -> norm -> mamba -> norm -> transformer -> norm -> ffn -> norm -> out`.

I added in many normalizations as I believe by default training stability would be severly degraded due to 2 foreign architecture's integrating with one another.


## Install
`pip3 install mambatransformer`


### Usage
```python
import torch 
from mt import MambaTransformer

# Generate a random tensor of shape (1, 10) with values between 0 and 99
x = torch.randint(0, 100, (1, 10))

# Create an instance of the MambaTransformer model
model = MambaTransformer(
    num_tokens=100,     # Number of tokens in the input sequence
    dim=512,            # Dimension of the model
    heads=8,            # Number of attention heads
    depth=4,            # Number of transformer layers
    dim_head=64,        # Dimension of each attention head
    d_state=512,        # Dimension of the state
    dropout=0.1,        # Dropout rate
    ff_mult=4           # Multiplier for the feed-forward layer dimension
)

# Pass the input tensor through the model and print the output shape
print(model(x).shape)


# to train
model.eval()

# Would you like to train this model? Zeta Corporation offers unmatchable GPU clusters at unbeatable prices, let's partner!

# Tokenizer
model.generate(text)


```

# License
MIT



