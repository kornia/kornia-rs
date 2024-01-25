import kornia_rs as K

import torch
import numpy as np

def test_smoke():
    # dumy test
    H, W, C = 2, 2, 3
    data = [i for i in range(H * W * C)]
    tensor = K.Tensor([H, W, C], data)
    assert tensor.shape == [H, W, C]
    assert len(data) == len(tensor.data)
    assert tensor.strides == [6, 3, 1]

def test_conversions():
    H, W, C = 2, 2, 3
    data = [i for i in range(H * W * C)]
    tensor = K.Tensor([H, W, C], data)

    # to dlpack / torch / numpy
    th_tensor = torch.utils.dlpack.from_dlpack(tensor)
    assert [x for x in th_tensor.shape] == tensor.shape

def test_conversions2():
    H, W, C = 2, 2, 3
    data = [i for i in range(H * W * C)]
    tensor = K.Tensor([H, W, C], data)

    # to dlpack / torch / numpy
    th_tensor = torch.utils.dlpack.from_dlpack(tensor)
    np_array = np.from_dlpack(tensor)
    np.testing.assert_array_equal(np_array, th_tensor.numpy())
