# Architecture

MiniGPT is a decoder-only language model. For token IDs shaped `[batch, time]`, learned token and position embeddings produce `[batch, time, d_model]`. Each pre-layer-normalized block applies causal multi-head self-attention and a GELU feed-forward network with residual connections. A final layer norm and tied vocabulary projection produce `[batch, time, vocab]` logits.

Each head forms Q, K, V projections, computes `softmax(QKᵀ / sqrt(head_dim) + causal_mask) V`, and concatenates heads. The lower-triangular mask prevents position `i` from using positions greater than `i`. Position embeddings supply order information. The output head shares its weight matrix with the token embedding.

Training shifts targets one character to the left: inputs `[t0,...,t(n-1)]` predict targets `[t1,...,tn]`. Cross entropy is averaged over batch and time. The model uses PyTorch autograd and AdamW.
