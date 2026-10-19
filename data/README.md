# Data

`tiny_corpus.txt` is an original, short educational fixture written for this repository. It is bundled for offline smoke testing and carries no external dataset attribution. It is not a representative language corpus and cannot support language-quality claims. For real experiments, use a text corpus whose license permits your intended use, document its source and license, and pass its local UTF-8 path to the training config.

`scripts/prepare_data.py --input PATH --output data/corpus.txt` copies a local text file. The tokenizer is fitted only to the training portion.

## Larger reproducible corpus: Alice's Adventures in Wonderland

Run `python scripts/prepare_data.py --alice --output data/corpus.txt` to fetch Lewis Carroll's *Alice's Adventures in Wonderland* (Project Gutenberg eBook 11). The script removes the Gutenberg wrapper and license preamble/closing text using the file's start/end markers. If Python cannot verify the website's certificate, download **Plain Text UTF-8** from the official book page in a browser, save it locally, and pass that file with `--input`; the script strips the wrapper in local Gutenberg files too. Do not disable TLS certificate checks. Source record: https://www.gutenberg.org/ebooks/11. Project Gutenberg identifies this work as public domain in the United States. Its terms ask users outside the U.S. to check their local law; India's Copyright Office states a general life-plus-60-year term for published literary works (https://copyright.gov.in/Copyright_Act_1957/chapter_v.html). Lewis Carroll died in 1898. Check local requirements before redistribution. The downloaded `data/corpus.txt` is ignored by Git; the command lets readers reproduce the experiment without committing a corpus copy.
