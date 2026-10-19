from minigpt.tokenizer import CharTokenizer, BPETokenizer, load_tokenizer

def test_roundtrip_and_unknown():
    t=CharTokenizer.train("cab"); assert t.decode(t.encode("cab"))=="cab"; assert t.encode("z")==[0]; assert t.decode([0])==""
def test_order_and_save_load(tmp_path):
    a=CharTokenizer.train("cba"); b=CharTokenizer.train("abc"); assert a.itos==b.itos
    path=tmp_path/"vocab.json"; a.save(path); assert CharTokenizer.load(path).itos==a.itos
def test_invalid_id():
    import pytest
    with pytest.raises(ValueError): CharTokenizer.train("a").decode([99])

def test_bpe_roundtrip_and_save_load(tmp_path):
    text = "Alice went to Wonderland. Alice said hello!"
    tokenizer = BPETokenizer.train(text, vocab_size=40)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) < len(text)
    path = tmp_path / "bpe.json"
    tokenizer.save(path)
    restored = load_tokenizer(path)
    assert restored.encode(text) == ids
    assert restored.decode(ids) == text

def test_bpe_is_deterministic_and_rejects_too_small_vocabulary():
    import pytest
    text = "the the there"
    a = BPETokenizer.train(text, 20)
    b = BPETokenizer.train(text, 20)
    assert a.itos == b.itos and a.merges == b.merges
    with pytest.raises(ValueError): BPETokenizer.train("abc", 3)

def test_bpe_can_roundtrip_tokens_split_across_segments():
    text = "Alice\n\nAlice went!"
    tokenizer = BPETokenizer.train(text, vocab_size=40)
    assert tokenizer.decode(tokenizer.encode(text)) == text
