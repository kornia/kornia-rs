import kornia_rs as K

def test_smoke():
    assert len(K.__version__) > 0
