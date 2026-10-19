# Interview preparation

1. **What does a decoder-only model predict?** The next token given the tokens before it.
2. **What do token embeddings do?** Map token IDs to learned vectors.
3. **Why add position embeddings?** Self-attention alone does not encode sequence order.
4. **What are Q, K, and V?** Query, key, and value projections used to calculate context mixing.
5. **Why scale attention scores by the square root of head size?** It keeps dot-product magnitudes stable before softmax.
6. **What does a causal mask do?** Prevents each position from using later tokens.
7. **Why use multiple heads?** Heads can represent different relationships in the context.
8. **What is a residual connection?** An identity path that helps information and gradients flow through a block.
9. **What does layer normalization do?** Normalizes hidden features to stabilize activations.
10. **What is cross entropy here?** The negative log probability assigned to the correct next token, averaged over tokens.
11. **How are targets aligned?** Inputs `[t0, ..., t(n-1)]` predict labels `[t1, ..., tn]`.
12. **What does backpropagation compute?** Loss gradients for trainable parameters using the chain rule.
13. **What does AdamW do?** Updates weights with adaptive gradient estimates and decoupled weight decay.
14. **What is perplexity?** Exponential of average token-level cross entropy under a fixed tokenizer and evaluation convention.
15. **Why reserve a test split?** It estimates performance after model and checkpoint selection without tuning on the test set.
16. **Why compare against an n-gram baseline?** It shows whether a neural model improves over a simpler conditional-frequency method.
17. **What does greedy decoding do?** Chooses the most probable next token at every generation step.
18. **How does generation handle long prompts?** It keeps the most recent tokens that fit the context window.
19. **Why doesn't this smoke experiment demonstrate language understanding?** The fixture is tiny and the run is short, so it can memorize patterns without learning broad language structure.
20. **What are character-tokenization tradeoffs?** Its vocabulary is transparent and small, but sequences are long and spelling must be learned character by character.
