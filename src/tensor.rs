#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::io::BufRead;

pub mod cv {

    use crate::dlpack_py::{cvtensor_to_dlpack, cvtensor_to_dltensor};
    use pyo3::prelude::*;

    fn get_strides_from_shape(shape: &[i64]) -> Vec<i64> {
        let mut strides = vec![0i64; shape.len()];

        let mut c = 1;
        strides[shape.len() - 1] = c;
        for i in (1..shape.len()).rev() {
            c *= shape[i];
            strides[i - 1] = c;
        }

        strides
    }

    #[pyclass]
    #[derive(Debug, Clone, PartialEq)]
    pub struct Tensor {
        #[pyo3(get)]
        pub shape: Vec<i64>,
        #[pyo3(get)]
        pub data: Vec<u8>,
        #[pyo3(get)]
        pub strides: Vec<i64>,
    }

    #[pymethods]
    impl Tensor {
        #[new]
        pub fn new(shape: Vec<i64>, data: Vec<u8>) -> Self {
            let strides = get_strides_from_shape(&shape);
            Tensor {
                shape,
                data,
                strides,
            }
        }

        #[pyo3(name = "__dlpack__")]
        pub fn dlpack_py(&self, py: Python) -> PyResult<PyObject> {
            cvtensor_to_dlpack(self, py)
        }

        #[pyo3(name = "__dlpack_device__")]
        pub fn dlpack_device_py(&self) -> (i32, i32) {
            let dl_tensor = cvtensor_to_dltensor(self);
            (
                dl_tensor.device.device_type as i32,
                dl_tensor.device.device_id,
            )
        }
    }

    impl Tensor {
        pub fn new_with_strides(shape: Vec<i64>, data: Vec<u8>, strides: Vec<i64>) -> Self {
            Tensor {
                shape,
                data,
                strides,
            }
        }
    }
} // namespace cv




fn add_simd(a: &cv::Tensor, b: &cv::Tensor) -> cv::Tensor {
    assert_eq!(a.shape, b.shape, "Tensor shapes must match");

    let mut data = vec![0u8; a.data.len()];

    unsafe {
        for i in (0..a.data.len()).step_by(16) {
            let a_chunk = _mm_loadu_si128(a.data.as_ptr().add(i) as *const __m128i);
            let b_chunk = _mm_loadu_si128(b.data.as_ptr().add(i) as *const __m128i);
            let sum = _mm_add_epi8(a_chunk, b_chunk);
            _mm_storeu_si128(data.as_mut_ptr().add(i) as *mut __m128i, sum);
        }
    }

    cv::Tensor {
        shape: a.shape.clone(),
        data,
        strides: a.strides.clone(),
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn generate_tensort(shape: Vec<i64>) -> cv::Tensor {
        let mut data = vec![0u8; shape.iter().product::<i64>() as usize];
        for i in 0..data.len() {
            data[i] = i as u8;
        }
        cv::Tensor::new(shape, data)
    }

    #[test]
    fn test_constructor() {
        let shape: Vec<i64> = vec![1, 1, 2, 2];
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let x: cv::Tensor = cv::Tensor::new(shape, data);
        assert_eq!(x.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_constructor_with_strides() {
        let shape: Vec<i64> = vec![1, 1, 2, 2];
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let strides: Vec<i64> = vec![4, 4, 2, 1];
        let x: cv::Tensor = cv::Tensor::new_with_strides(shape, data, strides);
        assert_eq!(x.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_add_simd() {
        let x = generate_tensort(vec![32, 128, 32, 32]);
        let y = generate_tensort(vec![32, 128, 32, 32]);
        let z = add_simd(&x, &y);
        assert_eq!(z.data[0], 0);
        assert_eq!(z.data[1], 2);
        assert_eq!(z.data[2], 4);
    }

}