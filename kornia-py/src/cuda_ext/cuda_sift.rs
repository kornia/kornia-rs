//! Device-resident CUDA SIFT (`kornia_rs.imgproc.sift_cuda`).
//!
//! Takes an f32 single-channel device `Image` in 0..255 -- the reference's own
//! internal representation, so a caller comparing against it must not normalise
//! to 0..1 -- and returns keypoints and 128-D descriptors as numpy arrays.
//!
//! The plan object owns its scratch, so it is rebuilt whenever the image size
//! changes but reused across frames of the same size.

use super::*;
use kornia_imgproc::cuda::sift::{
    FirstOctave, SiftCuda, SiftCudaConfig, SiftKeypoint, SiftMatcher, DESCR_LEN,
};
use numpy::PyArray2;

/// Everything that changes a plan's shape. Floats are compared by bits: these
/// come straight from the caller and are only ever tested for equality, so NaN's
/// `!=` itself would silently defeat the cache.
#[derive(PartialEq, Eq)]
pub(crate) struct PlanKey {
    ordinal: usize,
    width: usize,
    height: usize,
    n_features: usize,
    n_octave_layers: usize,
    contrast_bits: u64,
    edge_bits: u64,
    sigma_bits: u64,
    max_keypoints: usize,
    first_octave: FirstOctave,
    max_octaves: usize,
}

/// A plan and the shape it was built for. `None` until the first call.
pub(crate) type PlanSlot = Option<(PlanKey, SiftCuda)>;

/// Detect, orient and describe against a caller-owned plan.
///
/// The plan holds every scratch buffer in the pipeline — roughly a dozen
/// full-resolution planes — so it is rebuilt only when something that changes
/// its shape changes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sift_cuda_with_plan<'py>(
    py: Python<'py>,
    img: &PyImageApi,
    slot: &mut PlanSlot,
    n_features: usize,
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
            "sift: expected a device Image; create one with \
             Image.from_numpy(a).to_cuda(stream)",
        )
    })?;
    let Inner::F32C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "sift: expected an f32 single-channel device image in 0..255, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    if n_octave_layers == 0 {
        return Err(PyValueError::new_err(
            "sift: n_octave_layers must be non-zero",
        ));
    }
    if max_keypoints == 0 {
        return Err(PyValueError::new_err(
            "sift: max_keypoints must be non-zero",
        ));
    }
    let stream = source_stream(src)?;
    let ctx = stream.context();
    let size = src.size();

    let cfg = SiftCudaConfig {
        n_features,
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

    let key = PlanKey {
        ordinal: ctx.ordinal(),
        width: size.width,
        height: size.height,
        n_features,
        n_octave_layers,
        contrast_bits: contrast_threshold.to_bits(),
        edge_bits: edge_threshold.to_bits(),
        sigma_bits: sigma.to_bits(),
        max_keypoints,
        first_octave,
        max_octaves,
    };
    if slot.as_ref().map(|(k, _)| k) != Some(&key) {
        let plan = SiftCuda::new(
            &ctx,
            &stream,
            size.width,
            size.height,
            cfg,
            first_octave,
            max_octaves,
        )
        .map_err(err)?;
        *slot = Some((key, plan));
    }
    let (_, plan) = slot.as_mut().expect("plan just installed");
    let d_src = src
        .0
        .as_cudaslice()
        .ok_or_else(|| PyValueError::new_err("sift: device image has no typed f32 storage"))?;
    let feats = plan.detect_and_compute(&ctx, &stream, d_src).map_err(err)?;

    let kp_arr = keypoints_to_numpy(py, &feats.keypoints)?;
    let desc_arr = PyArray2::from_vec2(
        py,
        &feats
            .descriptors
            .chunks(DESCR_LEN)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((kp_arr, desc_arr))
}

/// The f32 single-channel device storage behind a Python `Image`, or a typed
/// error naming what was passed instead.
fn device_f32c1(img: &PyImageApi) -> PyResult<&Image<f32, 1>> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "sift: expected a device Image; create one with \
             Image.from_numpy(a).to_cuda(stream)",
        )
    })?;
    match dev {
        Inner::F32C1(src) => Ok(src),
        _ => Err(PyValueError::new_err(format!(
            "sift: expected an f32 single-channel device image in 0..255, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        ))),
    }
}

/// `(N, 6)` of `x, y, size, angle, response, octave`.
fn keypoints_to_numpy<'py>(
    py: Python<'py>,
    kps: &[SiftKeypoint],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows: Vec<Vec<f32>> = kps
        .iter()
        .map(|k| vec![k.x, k.y, k.size, k.angle, k.response, k.octave as f32])
        .collect();
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Detect in both images and match, without the descriptors ever leaving the
/// device.
///
/// The first image's descriptors are copied device-to-device into `store`
/// because the plan has one output buffer and the second detection overwrites
/// it. Only the keypoints and the surviving pairs come back to the host.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sift_match<'py>(
    py: Python<'py>,
    img_a: &PyImageApi,
    img_b: &PyImageApi,
    slot: &mut PlanSlot,
    store: &mut MatchStore,
    ratio: f32,
    cross_check: bool,
    n_features: usize,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i32>>,
)> {
    let params = (
        n_features,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
        upsample,
        max_octaves,
    );
    let (kps_a, na, stream, ctx) = detect_device(img_a, slot, params)?;
    {
        // Stash image A's descriptors before B's run overwrites the plan's
        // output buffer.
        let (_, plan) = slot.as_mut().expect("plan present after detect");
        store.ensure(&stream, max_keypoints * 4)?;
        if na > 0 {
            let src = plan.descriptors_device().slice(0..na * DESCR_LEN);
            let mut dst = store
                .desc_a
                .as_mut()
                .expect("just ensured")
                .slice_mut(0..na * DESCR_LEN);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    let (kps_b, nb, _, _) = detect_device(img_b, slot, params)?;

    let pairs = if na == 0 || nb == 0 {
        Vec::new()
    } else {
        let (_, plan) = slot.as_mut().expect("plan present after detect");
        let matcher = store.matcher.as_mut().expect("ensured");
        let a = store
            .desc_a
            .as_ref()
            .expect("ensured")
            .slice(0..na * DESCR_LEN);
        let b = plan.descriptors_device().slice(0..nb * DESCR_LEN);
        matcher
            .match_descriptors(&ctx, &stream, &a, na, &b, nb, ratio, cross_check)
            .map_err(err)?
    };

    let pair_rows: Vec<Vec<i32>> = pairs.iter().map(|p| vec![p[0], p[1]]).collect();
    let pair_arr =
        PyArray2::from_vec2(py, &pair_rows).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        keypoints_to_numpy(py, &kps_a)?,
        keypoints_to_numpy(py, &kps_b)?,
        pair_arr,
    ))
}

type DetectParams = (usize, usize, f64, f64, f64, usize, bool, usize);

/// Run detection, leaving descriptors in the plan's device buffer.
fn detect_device(
    img: &PyImageApi,
    slot: &mut PlanSlot,
    p: DetectParams,
) -> PyResult<(
    Vec<SiftKeypoint>,
    usize,
    std::sync::Arc<cudarc::driver::CudaStream>,
    std::sync::Arc<cudarc::driver::CudaContext>,
)> {
    let (n_features, n_octave_layers, contrast, edge, sigma, max_kp, upsample, max_oct) = p;
    let src = device_f32c1(img)?;
    let stream = source_stream(src)?;
    let ctx = stream.context();
    let size = src.size();
    let cfg = SiftCudaConfig {
        n_features,
        n_octave_layers,
        contrast_threshold: contrast,
        edge_threshold: edge,
        sigma,
        max_keypoints: max_kp,
    };
    let first_octave = if upsample {
        FirstOctave::Double
    } else {
        FirstOctave::Native
    };
    let max_octaves = if max_oct == 0 { usize::MAX } else { max_oct };
    let key = PlanKey {
        ordinal: ctx.ordinal(),
        width: size.width,
        height: size.height,
        n_features,
        n_octave_layers,
        contrast_bits: contrast.to_bits(),
        edge_bits: edge.to_bits(),
        sigma_bits: sigma.to_bits(),
        max_keypoints: max_kp,
        first_octave,
        max_octaves,
    };
    if slot.as_ref().map(|(k, _)| k) != Some(&key) {
        let plan = SiftCuda::new(
            &ctx,
            &stream,
            size.width,
            size.height,
            cfg,
            first_octave,
            max_octaves,
        )
        .map_err(err)?;
        *slot = Some((key, plan));
    }
    let (_, plan) = slot.as_mut().expect("plan just installed");
    let d_src = src
        .0
        .as_cudaslice()
        .ok_or_else(|| PyValueError::new_err("sift: device image has no typed f32 storage"))?;
    let kps = plan
        .detect_and_compute_device(&ctx, &stream, d_src)
        .map_err(err)?;
    let n = plan.descriptor_count();
    Ok((kps, n, stream.clone(), ctx.clone()))
}

/// Descriptor stash and matcher scratch, owned by the Python `Sift` object.
#[derive(Default)]
pub(crate) struct MatchStore {
    desc_a: Option<cudarc::driver::CudaSlice<f32>>,
    matcher: Option<SiftMatcher>,
    cap: usize,
}

impl MatchStore {
    fn ensure(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        cap: usize,
    ) -> PyResult<()> {
        if self.cap >= cap && self.desc_a.is_some() {
            return Ok(());
        }
        self.desc_a = Some(
            stream
                .alloc_zeros::<f32>(cap * DESCR_LEN)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        self.matcher = Some(SiftMatcher::new(stream, cap).map_err(err)?);
        self.cap = cap;
        Ok(())
    }
}
