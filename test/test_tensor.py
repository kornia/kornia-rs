import kornia_rs as K
from kornia_rs import Tensor as cvTensor

import torch

def test_smoke():
    # dumy test
    H, W, C = 2, 2, 3
    data = [i for i in range(H * W * C)]
    cv_tensor = cvTensor([H, W, C], data)
    assert cv_tensor.shape == [H, W, C]
    assert len(data) == len(cv_tensor.data)
    assert cv_tensor.strides == [6, 3, 1]

    # to dlpack / torch
    dlpack = K.cvtensor_to_dlpack(cv_tensor)
    th_tensor = torch.utils.dlpack.from_dlpack(dlpack)
    assert [x for x in th_tensor.shape] == cv_tensor.shape
