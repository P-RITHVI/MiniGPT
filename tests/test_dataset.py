import pytest
from minigpt.dataset import NextTokenDataset, random_batch, split_text
from scripts.prepare_data import extract_gutenberg_body

def test_shift_and_shape():
    d=NextTokenDataset([1,2,3,4,5],3); x,y=d[0]; assert x.tolist()==[1,2,3]; assert y.tolist()==[2,3,4]; assert len(d)==2
    bx,by=random_batch(d,4); assert bx.shape==by.shape==(4,3)
def test_short_and_split_isolation():
    assert len(NextTokenDataset([1,2],2))==0
    a,b,c=split_text("abcdefghij",.6,.2); assert a=="abcdef" and b=="gh" and c=="ij"
    with pytest.raises(ValueError): random_batch(NextTokenDataset([1],4),1)

def test_gutenberg_wrapper_removed():
    source="header\n*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nbody text\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nfooter"
    assert extract_gutenberg_body(source)=="body text"
