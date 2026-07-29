//! Device-resident CUDA SIFT, behind `kornia_rs.imgproc.Sift`.
//!
//! Takes an f32 single-channel device `Image` in 0..255 -- the reference's own
//! internal representation, so a caller comparing against it must not normalise
//! to 0..1 -- and returns host keypoints plus a **device** descriptor `Tensor`.
//!
//! The descriptors staying on device is the whole point: a 2515x128 block is
//! 1.3 MB, and moving it to the host costs more than the detection that produced
//! it. `Sift.match` takes those tensors straight back, so a frame-to-frame match
//! never touches host memory.
//!
//! The plan object owns its scratch, so it is rebuilt whenever the image size
//! changes but reused across frames of the same size.

use super::*;
use kornia_imgproc::cuda::sift::{FirstOctave, SiftCuda, SiftCudaConfig, SiftMatcher, DESCR_LEN};
use numpy::PyArray2;

/// Everything that changes a plan's shape, plus the one flag that does not.
///
/// Passed as a struct rather than nine positional arguments because the call
/// site had four `usize`s and three `f64`s in a row — a reordering that still
/// compiles and silently detects with the wrong thresholds.
#[derive(Clone, Copy)]
pub(crate) struct DetectParams {
    pub n_features: usize,
    pub n_octave_layers: usize,
    /// Selects a kernel, not a buffer size, so it is deliberately absent from
    /// `PlanKey`: keying on it would reallocate a dozen full-resolution device
    /// planes every time the flag flipped. `set_fast_descriptor` applies it on
    /// every call instead.
    pub fast_descriptor: bool,
    pub contrast_threshold: f64,
    pub edge_threshold: f64,
    pub sigma: f64,
    pub max_keypoints: usize,
    pub upsample: bool,
    pub max_octaves: usize,
}

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
///
/// `SiftCuda::detect_and_compute` returns an OWNED device descriptor buffer
/// (the final gather writes each frame into a fresh allocation), so the
/// `Tensor` takes it over zero-copy — no defensive D2D exists on this path.
pub(crate) fn sift_cuda_with_plan(
    py: Python<'_>,
    img: &PyImageApi,
    slot: &mut PlanSlot,
    p: DetectParams,
) -> crate::sift::DetectOut {
    let src = device_f32c1(img)?;
    let (stream, ctx) = ensure_plan(src, slot, p)?;
    let (_, plan) = slot.as_mut().expect("plan just installed");
    plan.set_fast_descriptor(p.fast_descriptor);
    let feats = plan.detect_and_compute(&ctx, &stream, src).map_err(err)?;
    let n = feats.len();

    let inner = if n == 0 {
        // No descriptor bytes exist; a host (0, 128) tensor states that
        // honestly. (The plan's returned device allocation is a 1-element
        // placeholder in this case — CUDA does not promise zero-byte
        // allocations — and it is simply dropped.)
        TensorInnerEnum::F32R2(kornia_tensor::Tensor::<f32, 2>::zeros([0, DESCR_LEN]))
    } else {
        TensorInnerEnum::F32R2(kornia_tensor::Tensor::from_cudaslice(
            feats.descriptors,
            [n, DESCR_LEN],
            stream.clone(),
        ))
    };
    let desc = PyTensor {
        inner: Arc::new(inner),
    }
    .into_pyobject(py)?
    .into_any()
    .unbind();
    Ok((crate::sift::keypoints_to_list(&feats.keypoints), desc))
}

/// Install a plan for `src` if the current one does not match, returning the
/// image's stream and context.
fn ensure_plan(
    src: &Image<f32, 1>,
    slot: &mut PlanSlot,
    p: DetectParams,
) -> PyResult<(
    Arc<cudarc::driver::CudaStream>,
    Arc<cudarc::driver::CudaContext>,
)> {
    if p.n_octave_layers == 0 {
        return Err(PyValueError::new_err(
            "sift: n_octave_layers must be non-zero",
        ));
    }
    if p.max_keypoints == 0 {
        return Err(PyValueError::new_err(
            "sift: max_keypoints must be non-zero",
        ));
    }
    let stream = source_stream(src)?;
    let ctx = stream.context().clone();
    let size = src.size();
    let cfg = SiftCudaConfig {
        n_features: p.n_features,
        n_octave_layers: p.n_octave_layers,
        contrast_threshold: p.contrast_threshold,
        edge_threshold: p.edge_threshold,
        sigma: p.sigma,
        max_keypoints: p.max_keypoints,
    };
    let first_octave = if p.upsample {
        FirstOctave::Double
    } else {
        FirstOctave::Native
    };
    let max_octaves = if p.max_octaves == 0 {
        usize::MAX
    } else {
        p.max_octaves
    };
    let key = PlanKey {
        ordinal: ctx.ordinal(),
        width: size.width,
        height: size.height,
        n_features: p.n_features,
        n_octave_layers: p.n_octave_layers,
        contrast_bits: p.contrast_threshold.to_bits(),
        edge_bits: p.edge_threshold.to_bits(),
        sigma_bits: p.sigma.to_bits(),
        max_keypoints: p.max_keypoints,
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

// ── Matching ─────────────────────────────────────────────────────────────────

/// The `(n, slice, stream)` behind a descriptor `Tensor`, or an error naming
/// what is wrong with it.
///
/// Validates the trailing dimension rather than trusting it: the matcher derives
/// its row count from the buffer length, so a `(1, 1, n, 64)` tensor would be
/// read as half as many 128-D rows instead of being rejected.
fn descriptor_block<'a>(
    t: &'a PyTensor,
    who: &str,
) -> PyResult<(
    usize,
    &'a cudarc::driver::CudaSlice<f32>,
    &'a Arc<cudarc::driver::CudaStream>,
)> {
    let TensorInnerEnum::F32R2(inner) = &*t.inner else {
        return Err(PyValueError::new_err(format!(
            "Sift.match: {who} must be a rank-2 float32 Tensor of shape \
             (N, {DESCR_LEN}); pass the descriptors from `detect_and_compute` \
             unmodified"
        )));
    };
    let shape = inner.shape;
    if shape[1] != DESCR_LEN {
        return Err(PyValueError::new_err(format!(
            "Sift.match: {who} has a trailing dimension of {}, expected {DESCR_LEN}; \
             pass the descriptors from `detect_and_compute` unmodified",
            shape[1]
        )));
    }
    let (slice, stream) = inner
        .as_cudaslice()
        .zip(inner.cuda_stream())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Sift.match: {who} is not device-resident; detect from a device \
                 Image to get device descriptors"
            ))
        })?;
    Ok((shape[0], slice, stream))
}

/// Match two device descriptor blocks, without either leaving the device.
///
/// Only the surviving pair indices come back to the host.
pub(crate) fn sift_match_device<'py>(
    py: Python<'py>,
    a: &PyTensor,
    b: &PyTensor,
    store: &mut MatchStore,
    max_keypoints: usize,
    ratio: f32,
    cross_check: bool,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    // An empty side has no pairs and no buffers to inspect, so answer before
    // demanding device residency — an empty detection yields a host tensor.
    // Spelled out rather than via `tdispatch!`: that macro is defined after this
    // module's declaration in `mod.rs`, so it is not in textual scope here.
    let empty = |t: &PyTensor| match &*t.inner {
        TensorInnerEnum::F32R2(x) => x.shape[0] == 0,
        // Not a descriptor block at all; `descriptor_block` below rejects it
        // with a message naming what it wanted.
        _ => false,
    };
    if empty(a) || empty(b) {
        // `(0, 2)` rather than `(0, 0)`, so a caller can still slice a column.
        return crate::pyutils::rows_to_numpy(py, Vec::<i32>::new(), 2);
    }

    let (na, da, stream) = descriptor_block(a, "descriptors_a")?;
    let (nb, db, stream_b) = descriptor_block(b, "descriptors_b")?;
    // Both blocks and every matcher launch are ordered on A's stream. If B was
    // produced on a different one there is no ordering edge between the kernel
    // that wrote B and the match that reads it, so refuse rather than race.
    if !Arc::ptr_eq(stream, stream_b) {
        return Err(PyValueError::new_err(
            "Sift.match: both descriptor blocks must be on the same CUDA stream",
        ));
    }
    let ctx = stream.context().clone();

    // The matcher's scratch is indexed by keypoint, so it must cover the larger
    // of the two blocks — which `max_keypoints` bounds, but a caller-supplied
    // tensor need not obey.
    store.ensure(stream, na.max(nb).max(max_keypoints))?;
    let matcher = store.matcher.as_mut().expect("just ensured");
    let pairs = matcher
        .match_descriptors(
            &ctx,
            stream,
            &da.slice(0..na * DESCR_LEN),
            na,
            &db.slice(0..nb * DESCR_LEN),
            nb,
            ratio,
            cross_check,
        )
        .map_err(err)?;

    let flat: Vec<i32> = pairs.iter().flat_map(|p| [p[0], p[1]]).collect();
    crate::pyutils::rows_to_numpy(py, flat, 2)
}

/// Matcher scratch, owned by the Python `Sift` object so it is allocated once
/// rather than per matched frame.
#[derive(Default)]
pub(crate) struct MatchStore {
    matcher: Option<SiftMatcher>,
    cap: usize,
}

impl MatchStore {
    fn ensure(&mut self, stream: &Arc<cudarc::driver::CudaStream>, cap: usize) -> PyResult<()> {
        // `cap` is checked alongside `is_some` rather than instead of it: if a
        // previous call failed to build the matcher, `cap` was never updated,
        // and a later call for a SMALLER cap would short-circuit here and leave
        // `matcher` None for the `expect` above.
        if self.cap >= cap && self.matcher.is_some() {
            return Ok(());
        }
        self.matcher = Some(SiftMatcher::new(stream, cap).map_err(err)?);
        self.cap = cap;
        Ok(())
    }
}
