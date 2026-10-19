import torch
from minigpt.attention import CausalSelfAttention

def test_shape_finite_and_future_blocked():
    torch.manual_seed(3); m=CausalSelfAttention(16,4,0,8).eval(); x=torch.randn(2,5,16); y=m(x)
    altered=x.clone(); altered[:,4]+=100
    y2=m(altered); assert y.shape==(2,5,16) and torch.isfinite(y).all(); assert torch.allclose(y[:,:4],y2[:,:4],atol=1e-6)
def test_masks_and_bad_dimensions():
    import pytest
    m=CausalSelfAttention(8,2,0,4); assert m.causal_mask[:3,:3].tolist()==[[True,False,False],[True,True,False],[True,True,True]]
    with pytest.raises(ValueError): CausalSelfAttention(7,2)
