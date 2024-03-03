//! This module provides utilities to convert between PyTorch and DLPack tensors.
//!
//! NOTE: this is deprecated and will be removed in the future.
//!
use dlpack_rs as dlpack;
use kornia_rs::tensor::Tensor;

use pyo3::prelude::*;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";

// desctructor function for the python capsule
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const c_char) == 1 {
        // println!("Is an invalid capsule!");
        return;
    }

    // println!("PyCapsule destructor");

    let expected_name = CString::new("dltensor").unwrap();

    let current_name_ptr: *const c_char = pyo3::ffi::PyCapsule_GetName(capsule);
    let current_name = CStr::from_ptr(current_name_ptr);
    // println!("Expected Name: {:?}", expected_name);
    // println!("Current Name: {:?}", current_name);

    if current_name != expected_name.as_c_str() {
        return;
    }

    let managed: *mut dlpack::DLManagedTensor =
        pyo3::ffi::PyCapsule_GetPointer(capsule, current_name_ptr) as *mut dlpack::DLManagedTensor;

    if managed.is_null() {
        // println!("Invalid managed pointer");
        return;
    }

    if !managed.is_null() {
        (*managed).deleter.unwrap()(managed);
    }

    // println!("Delete by Python");
}

unsafe extern "C" fn dlpack_deleter(_x: *mut dlpack::DLManagedTensor) {
    // println!("DLManagedTensor deleter");

    //let ctx = (*x).manager_ctx as *mut Tensor;
    //ctx.drop_in_place();
    //(*x).dl_tensor.shape.drop_in_place();
    //(*x).dl_tensor.strides.drop_in_place();
    //x.drop_in_place();
}

pub fn cvtensor_to_dltensor(x: &Tensor) -> dlpack::DLTensor {
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

fn cvtensor_to_dlmtensor(x: &Tensor) -> dlpack::DLManagedTensor {
    // create dl tensor

    let dl_tensor_bx = Box::new(x);
    let dl_tensor: dlpack::DLTensor = cvtensor_to_dltensor(&dl_tensor_bx);

    // create dlpack managed tensor

    dlpack::DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(dl_tensor_bx) as *mut c_void,
        deleter: Some(dlpack_deleter),
    }
}

pub fn cvtensor_to_dlpack(x: &Tensor, py: Python) -> PyResult<PyObject> {
    // create the managed tensor
    let dlm_tensor: dlpack::DLManagedTensor = cvtensor_to_dlmtensor(x);
    let dlm_tensor_bx = Box::new(dlm_tensor);

    // create python capsule
    let capsule = unsafe {
        let ptr = pyo3::ffi::PyCapsule_New(
            Box::into_raw(dlm_tensor_bx) as *mut c_void,
            DLPACK_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(dlpack_capsule_destructor as pyo3::ffi::PyCapsule_Destructor),
        );
        PyObject::from_owned_ptr(py, ptr)
    };
    Ok(capsule)
}
