import numpy as np
import pytest

import kornia_rs


@pytest.mark.parametrize(
    ("vector_type", "dtype"),
    [
        (kornia_rs.algebra.Vec3F32, np.float32),
        (kornia_rs.algebra.Vec3F64, np.float64),
    ],
)
def test_vec3_constructor(vector_type, dtype):
    vector = vector_type(1.0, 2.0, 3.0)

    assert vector.x == 1.0
    assert vector.y == 2.0
    assert vector.z == 3.0

    array = vector.as_numpy()

    assert array.shape == (3,)
    assert array.dtype == dtype
    np.testing.assert_array_equal(array, np.array([1.0, 2.0, 3.0], dtype=dtype))


@pytest.mark.parametrize(
    ("vector_type", "dtype"),
    [
        (kornia_rs.algebra.Vec3F32, np.float32),
        (kornia_rs.algebra.Vec3F64, np.float64),
    ],
)
def test_vec3_numpy_roundtrip(vector_type, dtype):
    array = np.array([1.5, -2.0, 4.25], dtype=dtype)

    vector = vector_type.from_numpy(array)
    result = vector.as_numpy()

    assert result.dtype == dtype
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize(
    ("vector_type", "dtype"),
    [
        (kornia_rs.algebra.Vec3F32, np.float32),
        (kornia_rs.algebra.Vec3F64, np.float64),
    ],
)
def test_vec3_rejects_wrong_length(vector_type, dtype):
    array = np.array([1.0, 2.0], dtype=dtype)

    with pytest.raises(ValueError, match="expected an array with shape"):
        vector_type.from_numpy(array)


def test_vec3_f32_rejects_float64():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(TypeError):
        kornia_rs.algebra.Vec3F32.from_numpy(array)


def test_vec3_f64_rejects_float32():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(TypeError):
        kornia_rs.algebra.Vec3F64.from_numpy(array)
