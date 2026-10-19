import argparse
from pathlib import Path
import sys
import re
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALICE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

def extract_gutenberg_body(text):
    start = re.search(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    end = re.search(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not find Project Gutenberg start/end markers in downloaded text")
    return text[start.end():end.start()].strip()

def read_local_inputs(paths):
    documents = []
    for source in paths:
        source = Path(source)
        if not source.is_file(): raise SystemExit(f"Input file does not exist: {source}")
        text = source.read_text(encoding="utf-8-sig")
        if re.search(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK", text, re.I):
            text = extract_gutenberg_body(text)
        if text.strip(): documents.append(text.strip())
    if not documents: raise SystemExit("Input text is empty")
    return "\n\n".join(documents)

def main():
    p=argparse.ArgumentParser(description="Prepare a local UTF-8 text corpus")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",action="append",help="Path to a local UTF-8 text file; repeat to combine multiple sources")
    group.add_argument("--alice",action="store_true",help="Download Alice's Adventures in Wonderland from Project Gutenberg")
    p.add_argument("--output",default="data/corpus.txt")
    args=p.parse_args()
    if args.alice:
        request=Request(ALICE_URL,headers={"User-Agent":"MiniGPT educational project/0.1 (text corpus preparation)"})
        try:
            with urlopen(request,timeout=30) as response: raw=response.read()
        except Exception as exc:
            raise SystemExit(f"Could not download the corpus from Project Gutenberg: {exc}") from exc
        text=extract_gutenberg_body(raw.decode("utf-8-sig"))
    else: text=read_local_inputs(args.input)
    if not text.strip(): raise SystemExit("Input text is empty")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text,encoding="utf-8")
    print(f"Wrote {len(text)} characters to {out}")
if __name__=="__main__": main()
