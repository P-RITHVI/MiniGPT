from minigpt.tokenizer import CharTokenizer

def test_roundtrip_and_unknown():
    t=CharTokenizer.train("cab"); assert t.decode(t.encode("cab"))=="cab"; assert t.encode("z")==[0]; assert t.decode([0])==""
def test_order_and_save_load(tmp_path):
    a=CharTokenizer.train("cba"); b=CharTokenizer.train("abc"); assert a.itos==b.itos
    path=tmp_path/"vocab.json"; a.save(path); assert CharTokenizer.load(path).itos==a.itos
def test_invalid_id():
    import pytest
    with pytest.raises(ValueError): CharTokenizer.train("a").decode([99])
