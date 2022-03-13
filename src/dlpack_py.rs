use crate::dlpack;
use crate::tensor::cv;
use pyo3::prelude::*;
use std::ffi::{c_void, CStr, CString};

// desctructor function for the python capsule
unsafe extern "C" fn destructor(o: *mut pyo3::ffi::PyObject) {
    // println!("PyCapsule destructor");

    let name = CString::new("dltensor").unwrap();

    let ptr = pyo3::ffi::PyCapsule_GetName(o);
    let current_name = CStr::from_ptr(ptr);
    // println!("Expected Name: {:?}", name);
    // println!("Current Name: {:?}", current_name);

    if current_name != name.as_c_str() {
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut dlpack::DLManagedTensor;
    (*ptr).deleter.unwrap()(ptr);

    // println!("Delete by Python");
}

pub fn cvtensor_to_dltensor(x: &cv::Tensor) -> dlpack::DLTensor {
    dlpack::DLTensor {
        data: x.data.as_ptr() as *mut c_void,
        device: dlpack::DLDevice {
            device_type: dlpack::DLDeviceType_kDLCPU,
            device_id: 0,
        },
        ndim: x.shape.len() as i32,
        dtype: dlpack::DLDataType {
            code: dlpack::DLDataTypeCode_kDLUInt as u8,
            bits: 8,
            lanes: 1,
        },
        shape: x.shape.as_ptr() as *mut i64,
        strides: x.strides.as_ptr() as *mut i64,
        byte_offset: 0,
    }
}

#[pyfunction]
pub fn cvtensor_to_dlpack(x: &cv::Tensor) -> PyResult<*mut pyo3::ffi::PyObject> {
    let dlm_tensor: dlpack::DLManagedTensor = x.to_dlpack();
    let dlm_tensor_bx = Box::new(dlm_tensor);

    let name = CString::new("dltensor").unwrap();

    // create python capsule
    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            Box::into_raw(dlm_tensor_bx) as *mut c_void,
            name.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };
    std::mem::forget(name);
    Ok(ptr)
}
