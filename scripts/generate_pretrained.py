"""Generate text with a small, publicly available pretrained GPT-2 model.

This is a separate inference path from MiniGPT. It does not train or modify
MiniGPT checkpoints; Transformers downloads the selected pretrained model the
first time it is used and caches it for later runs.
"""

from __future__ import annotations

import argparse
import sys


DEFAULT_MODEL = "distilbert/distilgpt2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate more fluent text with a pretrained GPT-2 model."
    )
    parser.add_argument("--prompt", required=True, help="Text to continue")
    parser.add_argument("--tokens", type=int, default=100, help="Maximum new tokens")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Keep this many likely next tokens; use 0 to disable")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold (0 < value <= 1)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3,
                        help="Block repeated n-grams; use 0 to disable")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must contain text")
    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if args.top_k < 0:
        parser.error("--top-k must be 0 or greater")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if args.no_repeat_ngram_size < 0:
        parser.error("--no-repeat-ngram-size must be 0 or greater")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "Missing pretrained-generation dependencies. From the project root, run:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but PyTorch cannot access a CUDA device")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:
        print(f"Could not load model '{args.model}': {exc}", file=sys.stderr)
        print(
            "Check that you have internet access on the first run. The model files are cached "
            "after a successful download. If the error mentions a certificate, fix the "
            "Python/Windows certificate configuration; do not disable TLS verification.",
            file=sys.stderr,
        )
        return 1

    model.to(device)
    model.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", None)
    if context_limit:
        input_ids = input_ids[:, -(context_limit - 1):]
        max_new_tokens = min(args.tokens, context_limit - input_ids.shape[1])
    else:
        max_new_tokens = args.tokens
    if max_new_tokens < 1:
        parser.error("The model context window leaves no room for generated tokens")
    input_ids = input_ids.to(device)

    generation_options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.no_repeat_ngram_size:
        generation_options["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generation_options)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
