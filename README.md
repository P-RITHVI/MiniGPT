# MiniGPT

MiniGPT is a small, inspectable decoder-only Transformer language model implemented with PyTorch tensor operations, autograd, and AdamW. It is for learning the mechanics of next-token prediction, not a pretrained model or production assistant.

## Architecture

```mermaid
flowchart LR
  A[Character IDs] --> B[Token + position embeddings]
  B --> C[Pre-LN causal multi-head attention]
  C --> D[Residual + pre-LN GELU MLP]
  D --> E[Repeated Transformer blocks]
  E --> F[Final layer norm]
  F --> G[Tied vocabulary projection]
  G --> H[Next-character logits]
```

The default deterministic character vocabulary is built from training text only. The optional byte-pair tokenizer learns frequent adjacent pieces from training text, reducing sequence length while preserving the source text. Each training example shifts targets by one token. Causal attention masks future tokens, uses scaled dot products, and combines multiple heads. See [architecture](docs/architecture.md), [training](docs/training.md), [evaluation](docs/evaluation.md), and [limitations](docs/limitations.md).

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_smoke_test.py
```

The smoke test trains five CPU steps on the bundled educational fixture and writes local files to `artifacts/smoke/`. It is not a quality experiment.

## Data and training

The bundled `data/tiny_corpus.txt` is original educational text included for offline execution. For a larger reproducible experiment, download *Alice's Adventures in Wonderland*, Project Gutenberg eBook 11. Its catalog lists the work as public domain in the USA. Project Gutenberg advises readers elsewhere to check local law; India's Copyright Office describes the general life-plus-60-year term for published literary works. See [data provenance and terms](data/README.md). The download command requires internet access:

```bat
python scripts\prepare_data.py --alice --output data\corpus.txt
```

If Python reports a certificate error, download the **Plain Text UTF-8** file from the [official book page](https://www.gutenberg.org/ebooks/11) in your browser, save it as `data\alice_raw.txt`, then run `python scripts\prepare_data.py --input data\alice_raw.txt --output data\corpus.txt`. The script recognizes and removes the Gutenberg wrapper in that local file too. Do not disable certificate verification.

Alternatively, prepare a local UTF-8 text file:

```powershell
python scripts/prepare_data.py --input PATH_TO_TEXT_FILE --output data/corpus.txt
```

Train directly against the prepared corpus (the default config and offline smoke flow remain unchanged), then evaluate the resulting checkpoint:

```powershell
python scripts/train.py --config configs/default.json --data data/corpus.txt --output-dir artifacts/alice
python scripts/train.py --config configs/default.json --data data/corpus.txt --output-dir artifacts/alice --steps 10
python scripts/evaluate.py --checkpoint artifacts/alice/minigpt.pt
python scripts/generate.py --checkpoint artifacts/alice/minigpt.pt --prompt "Language models" --greedy
```

For a stronger experiment, use the optional BPE configuration. It learns up to 512 subword pieces from the training split, uses a 128-token context, a 3-layer 96-dimensional model, and trains for 3,000 steps. CPU training took about ten minutes on the tested environment. Training selects the best validation checkpoint. Alice alone is still a small educational corpus, not enough to train a general-purpose language model:

```bat
python scripts/train.py --config configs/alice_bpe.json --data data/corpus.txt --output-dir artifacts/alice_bpe
python scripts/evaluate.py --checkpoint artifacts/alice_bpe/minigpt.pt
python scripts/generate.py --checkpoint artifacts/alice_bpe/minigpt.pt --prompt "Alice" --tokens 100 --temperature 0.8 --top-k 20 --seed 1337
```

For a more varied corpus, download the **Plain Text UTF-8** files from [Sherlock Holmes](https://www.gutenberg.org/ebooks/1661), [Tom Sawyer](https://www.gutenberg.org/ebooks/74), and [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342), save them as `data\sherlock_raw.txt`, `data\tom_sawyer_raw.txt`, and `data\pride_prejudice_raw.txt`, and combine with Alice:

```bat
python scripts\prepare_data.py --input data\alice_raw.txt --input data\sherlock_raw.txt --input data\tom_sawyer_raw.txt --input data\pride_prejudice_raw.txt --output data\corpus_multi.txt
python scripts\train.py --config configs\alice_bpe.json --data data\corpus_multi.txt --output-dir artifacts\multi_book_bpe
python scripts\evaluate.py --checkpoint artifacts\multi_book_bpe\minigpt.pt --data data\corpus_multi.txt
python scripts\generate.py --checkpoint artifacts\multi_book_bpe\minigpt.pt --prompt "Alice" --tokens 100 --temperature 0.8 --top-k 20
```

Project Gutenberg labels those books public domain in the USA; check local laws before use or redistribution. Its license explains terms for users outside the USA. See [data provenance and terms](data/README.md).

Generation truncates old context to the configured window, supports greedy decoding, temperature and top-k. Checkpoints and generated metrics are local and ignored by Git. Evaluation must be run on the unchanged corpus used during training to reproduce its split convention.

### Want more fluent generation?

The from-scratch MiniGPT is the educational model and remains available with the commands above. For more fluent text, install dependencies and use the separate pretrained DistilGPT2 generator. It downloads its model on the first run and needs internet access then; see [pretrained generation](docs/pretrained_generation.md) for setup, options, and limitations:

```bat
python -m pip install -r requirements.txt
python scripts\generate_pretrained.py --prompt "Alice was beginning to get very tired of sitting by her sister on the bank" --tokens 100 --temperature 0.8 --top-k 50 --seed 1337
```

Measured on the single Alice corpus in the CPU environment: token-level test loss 2.5673 (perplexity 13.03) versus the same-tokenization bigram baseline at 3.4539 (perplexity 31.62). The model has 376,224 parameters and took 619.7 seconds; its best sampled validation loss was 2.4848 at step 2,750. These token-level scores cannot be directly compared to the character-level experiment above. Fixed-seed outputs from three prompts are in [the sample log](docs/alice_bpe_samples.txt). They imitate some book-like punctuation and dialogue but still have spelling and grammar errors; this is not fluent text generation. The single-book corpus remains the main data limitation. The measured per-step history is tracked in [the BPE training CSV](docs/alice_bpe_training_history.csv); the checkpoint stays local and is not included in Git.

The four-book corpus contains 1,828,252 characters. On it, the 3,000-step BPE model reached test loss 2.5357 (perplexity 12.63), beating its token-level bigram baseline at 3.0804 (perplexity 21.77). Extending training to 6,000 steps improved test loss to 2.3073 (perplexity 10.05) against the same baseline. The larger run has 377,376 parameters and took 1,084.9 seconds; best validation loss was 2.3249 at step 5,500. These scores are specific to the four-book mixture and cannot be directly compared with the Alice-only test score. Fixed-seed samples are in the [3,000-step sample log](docs/multi_book_bpe_samples.txt) and [6,000-step sample log](docs/multi_book_bpe_6000_samples.txt). Longer training improved the loss but generation still contains malformed words and sentences; this project does not yet generate correct fluent text. The [3,000-step history](docs/multi_book_bpe_training_history.csv) and [6,000-step history](docs/multi_book_bpe_6000_training_history.csv) record the measured evaluations.

## Measured results

The bundled corpus is only a smoke fixture. The table includes the five-step functionality check, the user's original final-step run, and a rerun with the best-validation checkpoint selection now implemented. The selected model was chosen on validation loss at step 100 and evaluated on the held-out final split afterward. All results are fixture-specific and are not evidence of general language ability.

| Model | Train loss | Validation loss | Test loss | Perplexity | Parameters | Training time |
|---|---:|---:|---:|---:|---:|---:|
| MiniGPT (5-step smoke) | 3.5812 | 3.5131 | 3.5251 | 33.9564 | 15,104 | 0.053 s |
| MiniGPT (300 steps, best validation at step 100) | 2.2447* | 2.7452 | 2.5840 | 13.2500 | 106,816 | 19.65 s |
| MiniGPT (300 steps, previous final checkpoint) | 1.3839* | 3.5456 | 3.0402 | 20.9086 | 106,816 | 9.49 s |
| Add-one character bigram | N/A — not measured | N/A — not measured | 2.8374 | 17.0717 | N/A — not recorded | N/A — not recorded |

*Training loss is the final minibatch at the selected step, not an average over the entire training split. The previous final-step model overfit substantially. Best-validation selection improved the measured test loss and beat the baseline on this tiny fixture, which is not sufficient evidence of broad language-model quality. The selected model generated `Languagentin te te te tin ten te tion ton t an t te te t te te ten t`, still poor text. The five-step smoke generation was `Languageennnnnnnnnnn`. No corpus-scale experiment has been run. Current training writes a loss CSV and, when there are at least three evaluation points, a plot.

## Resource expectations

The default configuration is a 2-layer, 64-dimensional model with context length 64 and batch size 16. It runs on CPU; CUDA is optional but not configured by default. Dependencies are PyTorch, pytest, and matplotlib. PyTorch checkpoints should only be loaded from trusted sources.

## Repository layout

`minigpt/` contains the tokenizer, dataset, attention, Transformer, model, training, evaluation and checkpoint code. `scripts/` contains command-line entry points, `tests/` contains the automated suite, and `docs/` contains implementation notes. The license is MIT. The bundled fixture is original and has no external dataset attribution.

Interview questions and concise explanations are in [docs/interview_prep.md](docs/interview_prep.md).

## License and attribution

Code is released under the MIT License. The educational fixture was written for this project. External corpora are not bundled; users are responsible for documenting and complying with their selected dataset license.
