# Pretrained text generation

The `scripts/generate.py` command runs the small MiniGPT model trained from scratch in this project. It is useful for studying the training code, but it does not have enough data and compute to produce consistently grammatical prose.

For more coherent, instruction-following text, this project also includes `scripts/generate_pretrained.py`. By default, it loads `Qwen/Qwen2.5-0.5B-Instruct`, a separate, already-trained instruction model. Its weights are about 988 MB, so this is an additional download beyond DistilGPT2. The script applies the model's chat template automatically. It does not train or change the MiniGPT checkpoint. The Qwen model card lists Apache-2.0 as its license; review the model card and its terms before redistribution or other use.

From the project root in the VS Code terminal, with `.venv` active, install the dependencies:

```bat
python -m pip install -r requirements.txt
```

Then generate a continuation:

```bat
python scripts\generate_pretrained.py --prompt "Write a short, coherent fantasy scene about Alice following the White Rabbit into a curious garden. Keep the events consistent and use clear English." --tokens 160 --temperature 0.7 --top-k 40 --top-p 0.9 --seed 1337
```

On the first run, the script downloads the model and tokenizer from Hugging Face. It needs internet access, and later runs use the local cache. If a download fails with an SSL certificate error, fix the Python or Windows certificate configuration; do not turn off certificate verification.

The script runs on CPU by default, selects CUDA automatically when available, and accepts `--device cpu` or `--device cuda`. Sampling controls include `--temperature`, `--top-k`, `--top-p`, and `--seed`. The default 3-token repeat block and a mild repetition penalty help reduce loops, but generation is still probabilistic. A small model can still make mistakes or lose track of a story; this should improve instruction-following compared with base DistilGPT2, but it is not a guarantee of flawless writing. To use the older base model instead, pass `--model distilbert/distilgpt2`; that model has no chat template, so it receives the prompt as plain text. A pretrained model does not guarantee factual or copyright-safe output. MiniGPT's original training, evaluation, and generation commands remain available as before.

Model details and license: [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [DistilGPT2 model card](https://huggingface.co/distilbert/distilgpt2). Chat-template handling follows the [Transformers documentation](https://huggingface.co/docs/transformers/chat_templating). Transformers installation details: [Hugging Face documentation](https://huggingface.co/docs/transformers/en/installation).
