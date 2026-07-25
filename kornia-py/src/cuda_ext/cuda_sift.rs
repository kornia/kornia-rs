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
    fast_descriptor: bool,
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
    fast_descriptor: bool,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let params = (
        n_features,
        n_octave_layers,
        fast_descriptor,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
        upsample,
        max_octaves,
    );
    let src = device_f32c1(img)?;
    let (stream, ctx) = ensure_plan(src, slot, params)?;
    let (_, plan) = slot.as_mut().expect("plan just installed");
    plan.set_fast_descriptor(fast_descriptor);
    let d_src = src
        .0
        .as_cudaslice()
        .ok_or_else(|| PyValueError::new_err("sift: device image has no typed f32 storage"))?;
    let feats = plan.detect_and_compute(&ctx, &stream, d_src).map_err(err)?;

    let kp_arr = keypoints_to_numpy(py, &feats.keypoints)?;
    let desc_arr = rows_to_numpy(py, feats.descriptors, DESCR_LEN)?;
    Ok((kp_arr, desc_arr))
}

/// Reshape a flat row-major block into `(len / cols, cols)`.
///
/// `PyArray2::from_vec2` derives the column count from the first row, so an
/// empty result would come back as `(0, 0)` instead of `(0, cols)` and break
/// any caller that slices a column or stacks the array. Going through a flat
/// vector also skips the per-row `Vec` allocation and copy.
fn rows_to_numpy<T: numpy::Element>(
    py: Python<'_>,
    flat: Vec<T>,
    cols: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let rows = flat.len() / cols;
    numpy::PyArray1::from_vec(py, flat)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Install a plan for `src` if the current one does not match, returning the
/// image's stream and context.
///
/// Shared by both entry points so the key and the rebuild rule cannot drift.
fn ensure_plan(
    src: &Image<f32, 1>,
    slot: &mut PlanSlot,
    p: DetectParams,
) -> PyResult<(
    std::sync::Arc<cudarc::driver::CudaStream>,
    std::sync::Arc<cudarc::driver::CudaContext>,
)> {
    let (
        n_features,
        n_octave_layers,
        fast_descriptor,
        contrast,
        edge,
        sigma,
        max_kp,
        upsample,
        max_oct,
    ) = p;
    if n_octave_layers == 0 {
        return Err(PyValueError::new_err(
            "sift: n_octave_layers must be non-zero",
        ));
    }
    if max_kp == 0 {
        return Err(PyValueError::new_err(
            "sift: max_keypoints must be non-zero",
        ));
    }
    let stream = source_stream(src)?;
    let ctx = stream.context().clone();
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
        fast_descriptor,
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
    Ok((stream, ctx))
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

/// Columns in the keypoint array: `x, y, size, angle, response, octave`.
const KP_COLS: usize = 6;

/// `(N, 6)` of `x, y, size, angle, response, octave` — `(0, 6)` when empty.
fn keypoints_to_numpy<'py>(
    py: Python<'py>,
    kps: &[SiftKeypoint],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut flat = Vec::with_capacity(kps.len() * KP_COLS);
    for k in kps {
        flat.extend_from_slice(&[k.x, k.y, k.size, k.angle, k.response, k.octave as f32]);
    }
    rows_to_numpy(py, flat, KP_COLS)
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
    fast_descriptor: bool,
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
        fast_descriptor,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
        upsample,
        max_octaves,
    );
    let (kps_a, na, stream, ctx) = detect_device(img_a, slot, params)?;
    // Everything after this point — the descriptor stash, B's detection and the
    // matcher launches — is issued on A's stream. If B lives on a different one
    // there is no ordering edge between B's kernels and the match, so refuse
    // rather than race.
    if !std::sync::Arc::ptr_eq(&stream, &source_stream(device_f32c1(img_b)?)?) {
        return Err(PyValueError::new_err(
            "Sift.match: both images must be on the same CUDA stream",
        ));
    }
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

    let mut pair_flat = Vec::with_capacity(pairs.len() * 2);
    for p in &pairs {
        pair_flat.extend_from_slice(p);
    }
    // `(0, 2)` rather than `(0, 0)` when nothing matched.
    let pair_arr = rows_to_numpy(py, pair_flat, 2)?;
    Ok((
        keypoints_to_numpy(py, &kps_a)?,
        keypoints_to_numpy(py, &kps_b)?,
        pair_arr,
    ))
}

type DetectParams = (usize, usize, bool, f64, f64, f64, usize, bool, usize);

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
    let src = device_f32c1(img)?;
    let (stream, ctx) = ensure_plan(src, slot, p)?;
    let (_, plan) = slot.as_mut().expect("plan just installed");
    plan.set_fast_descriptor(p.2);
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
