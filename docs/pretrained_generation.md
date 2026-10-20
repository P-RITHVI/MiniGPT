# Pretrained text generation

The `scripts/generate.py` command runs the small MiniGPT model trained from scratch in this project. It is useful for studying the training code, but it does not have enough data and compute to produce consistently grammatical prose.

For more fluent continuations, this project also includes `scripts/generate_pretrained.py`. It loads `distilbert/distilgpt2`, a separate, already-trained GPT-2-family model. This path does not train or change the MiniGPT checkpoint. The model card lists Apache-2.0 as its license; review the model card and its terms before redistribution or other use.

From the project root in the VS Code terminal, with `.venv` active, install the dependencies:

```bat
python -m pip install -r requirements.txt
```

Then generate a continuation:

```bat
python scripts\generate_pretrained.py --prompt "Alice was beginning to get very tired of sitting by her sister on the bank" --tokens 100 --temperature 0.8 --top-k 50 --seed 1337
```

On the first run, the script downloads the model and tokenizer from Hugging Face. It needs internet access, and later runs use the local cache. If a download fails with an SSL certificate error, fix the Python or Windows certificate configuration; do not turn off certificate verification.

The script runs on CPU by default, selects CUDA automatically when available, and accepts `--device cpu` or `--device cuda`. Sampling controls include `--temperature`, `--top-k`, `--top-p`, and `--seed`. The default 3-token repeat block and a mild repetition penalty help reduce loops, but generation is still probabilistic. A pretrained model can make mistakes and does not guarantee factual or copyright-safe output. MiniGPT's original training, evaluation, and generation commands remain available as before.

Model details and license: [distilbert/distilgpt2 model card](https://huggingface.co/distilbert/distilgpt2). Transformers installation details: [Hugging Face documentation](https://huggingface.co/docs/transformers/en/installation).
