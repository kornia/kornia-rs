"""Device (CUDA) path of connected_components: label-identical to CPU."""

import numpy as np
import pytest

import kornia_rs
from kornia_rs import imgproc

from _cuda_helpers import dev as _dev

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="CUDA not available (no GPU or CPU-only wheel)",
)


@pytest.mark.parametrize("shape", [(48, 64), (43, 67)])
@pytest.mark.parametrize("conn", [4, 8])
def test_ccl_device_matches_cpu(shape, conn):
    rng = np.random.default_rng(3)
    img = ((rng.random(shape) < 0.4) * 255).astype(np.uint8)[..., None]
    n_cpu, lab_cpu = imgproc.connected_components(img, connectivity=conn)
    n_gpu, lab_dev = imgproc.connected_components(_dev(img), connectivity=conn)
    assert n_cpu == n_gpu
    assert str(np.asarray(lab_dev.to_numpy()).dtype) == "int32"
    np.testing.assert_array_equal(np.asarray(lab_dev.to_numpy()), np.asarray(lab_cpu))
