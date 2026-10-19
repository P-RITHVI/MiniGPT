# Evaluation

Cross entropy is the mean next-character negative log likelihood; perplexity is `exp(loss)` and is omitted if exponentiation would overflow. Validation is sampled during training. `scripts/evaluate.py` computes test loss from the final contiguous split after training. The BigramBaseline module implements an add-one-smoothed character bigram model as a reference, but no baseline metric is recorded until explicitly run. Generated text is qualitative evidence only. No results are claimed before execution.
