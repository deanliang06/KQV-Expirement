## U-Net Embedding Bottleneck

This variant keeps the same overall training setup as the root experiment, but changes the `Q` reconstruction path:

- encode the original `Q` matrix with a U-Net-style down path
- pool the deepest feature map into a learned embedding
- decode that embedding with a single linear layer back into a full `Q`-sized matrix

`K` and `V` remain unchanged.
