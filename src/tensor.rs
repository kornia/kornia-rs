pub mod cv {

    use pyo3::prelude::*;
    use crate::dlpack;
    use crate::dlpack_py::{cvtensor_to_dltensor, cvtensor_to_dlpack};
    use std::ffi::{c_void};

    unsafe extern "C" fn deleter(x: *mut dlpack::DLManagedTensor) {
        // println!("DLManagedTensor deleter");

        let ctx = (*x).manager_ctx as *mut Tensor;
        ctx.drop_in_place();
        (*x).dl_tensor.shape.drop_in_place();
        (*x).dl_tensor.strides.drop_in_place();
        x.drop_in_place();
    }

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
        pub fn to_dlpack_py(&self) -> PyResult<*mut pyo3::ffi::PyObject> {
            return cvtensor_to_dlpack(self);
        }

        #[pyo3(name = "__dlpack_device__")]
        pub fn to_dlpack_device_py(&self) -> (u32, i32) {
            let tensor_bx = Box::new(self);
            let dl_tensor = cvtensor_to_dltensor(&tensor_bx);
            (dl_tensor.device.device_type, dl_tensor.device.device_id)
        }
    }

    impl Tensor {
        pub fn to_dlpack(&self) -> dlpack::DLManagedTensor {
            // we need to clone to avoid race conditions
            // TODO: check how to avoid that
            let tensor_bx = Box::new(self.clone());
            let dl_tensor = cvtensor_to_dltensor(&tensor_bx);

            // create dlpack managed tensor
            let dlm_tensor = dlpack::DLManagedTensor {
                dl_tensor,
                manager_ctx: Box::into_raw(tensor_bx) as *mut c_void,
                deleter: Some(deleter),
            };
            dlm_tensor
        }
    }

} // namespace cv

// TODO(carlos): enable tests later
//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn add() {
//        let shape: Vec<usize> = vec![1, 1, 2, 2];
//        let data: Vec<u8> = (0..cv::cumprod(&shape)).map(|x| x as u8).collect();
//        let t1 = cv::Tensor::new(shape.clone(), data);
//        let t2 = t1.clone();
//        let t3 = t1.add(t2.clone());
//        let to_compare = cv::Tensor::new(shape.clone(), vec![0, 2, 4, 6]);
//        assert_eq!(t3, to_compare);
//    }
//}
