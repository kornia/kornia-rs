pub mod cv {

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
