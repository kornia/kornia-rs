//! Device-resident CUDA SIFT (`kornia_rs.imgproc.sift_cuda`).
//!
//! Takes an f32 single-channel device `Image` in 0..255 -- the reference's own
//! internal representation, so a caller comparing against it must not normalise
//! to 0..1 -- and returns keypoints and 128-D descriptors as numpy arrays.
//!
//! The plan object owns its scratch, so it is rebuilt whenever the image size
//! changes but reused across frames of the same size.

use super::*;
use kornia_imgproc::cuda::sift::{FirstOctave, SiftCuda, SiftCudaConfig};
use numpy::PyArray2;

/// Detect, orient and describe.
///
/// Returns `(keypoints, descriptors)` where `keypoints` is `(N, 6)` holding
/// `x, y, size, angle, response, octave` and `descriptors` is `(N, 128)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sift_cuda<'py>(
    py: Python<'py>,
    img: &PyImageApi,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "sift_cuda: expected a device Image; create one with \
             Image.from_numpy(a).to_cuda(stream)",
        )
    })?;
    let Inner::F32C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "sift_cuda: expected an f32 single-channel device image in 0..255, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    if n_octave_layers == 0 {
        return Err(PyValueError::new_err(
            "sift_cuda: n_octave_layers must be non-zero",
        ));
    }
    if max_keypoints == 0 {
        return Err(PyValueError::new_err(
            "sift_cuda: max_keypoints must be non-zero",
        ));
    }
    let stream = source_stream(src)?;
    let ctx = stream.context();
    let size = src.size();

    let cfg = SiftCudaConfig {
        n_features: 0,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
    };
    let first_octave = if upsample {
        FirstOctave::Double
    } else {
        FirstOctave::Native
    };
    let max_octaves = if max_octaves == 0 {
        usize::MAX
    } else {
        max_octaves
    };

    let mut plan = SiftCuda::new(
        &ctx,
        &stream,
        size.width,
        size.height,
        cfg,
        first_octave,
        max_octaves,
    )
    .map_err(err)?;
    let d_src = src
        .0
        .as_cudaslice()
        .ok_or_else(|| PyValueError::new_err("sift_cuda: device image has no typed f32 storage"))?;
    let feats = plan.detect_and_compute(&ctx, &stream, d_src).map_err(err)?;

    let n = feats.len();
    let mut kp = vec![0.0f32; n * 6];
    for (i, k) in feats.keypoints.iter().enumerate() {
        kp[i * 6] = k.x;
        kp[i * 6 + 1] = k.y;
        kp[i * 6 + 2] = k.size;
        kp[i * 6 + 3] = k.angle;
        kp[i * 6 + 4] = k.response;
        kp[i * 6 + 5] = k.octave as f32;
    }
    let kp_arr = PyArray2::from_vec2(py, &kp.chunks(6).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let desc_arr = PyArray2::from_vec2(
        py,
        &feats
            .descriptors
            .chunks(128)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((kp_arr, desc_arr))
}
