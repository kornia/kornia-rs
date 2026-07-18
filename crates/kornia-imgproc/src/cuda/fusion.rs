//! FKL-style kernel fusion engine.
//!
//! Composes per-op device snippets into ONE NVRTC kernel where data flows
//! through registers between stages — no intermediate global-memory round
//! trips. The design follows the Fused Kernel Library model (Amoros /
//! Nuñez / Peña, <https://github.com/Libraries-Openly-Fused>):
//!
//! - each stage splits into a host-side **build** step (user-friendly →
//!   kernel-friendly parameters, done once) and a device-side **exec**
//!   snippet (the hot path);
//! - all stages' kernel-friendly parameters are packed into a single POD
//!   blob passed as **`__grid_constant__ const`** — parameters read from
//!   the constant bank cost zero registers and broadcast warp-wide, so the
//!   register footprint of a chain depends on the exec bodies, not on how
//!   much state the stages carry;
//! - the composed kernel is compiled once per (pipeline shape) and cached.
//!
//! # Value flow
//!
//! v1 pipelines are **per-output-pixel** over a destination grid: a source
//! stage produces a `float3 v` for the output coordinate `(x, y)`, middle
//! stages map `v -> v`, and a sink stage writes `v` (or a reduction of it)
//! to the destination. Intermediates stay in f32 registers; fused output
//! is therefore NOT bit-equal to running the ops separately through u8
//! buffers — it is a documented, higher-precision contract of its own
//! (the same one the existing hand-fused preprocessor uses).
//!
//! # Parameter blob layout
//!
//! Each stage contributes 4-byte-aligned fields appended in stage order;
//! the generated CUDA source declares `struct FusedParams { unsigned int
//! raw[N]; }` and per-field accessor macros with baked offsets, so field
//! access compiles to a constant-bank read at a literal offset.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};
use kornia_tensor::CudaKernel;

use super::{make_config, try_compile_with_l1};

/// Error type for the fusion engine.
#[derive(Debug, thiserror::Error)]
pub enum FusionError {
    /// Kernel compilation or launch failure.
    #[error("fusion kernel compile/launch error: {0}")]
    Cuda(String),
    /// Pipeline shape invalid (no source, no sink, type mismatch...).
    #[error("invalid pipeline: {0}")]
    Pipeline(String),
    /// Parameter blob exceeded the fixed capacity.
    #[error("fused parameter blob exceeds {cap} bytes (needs {need})")]
    ParamsTooLarge {
        /// Fixed blob capacity in bytes.
        cap: usize,
        /// Bytes required by this pipeline.
        need: usize,
    },
}

/// Fixed capacity of the packed parameter blob (bytes). 256 B of constant
/// bank covers ~60 scalar fields — far beyond current pipelines; grow if a
/// stage ever carries more (a `ParamsTooLarge` error, never silent).
pub const PARAM_BLOB_BYTES: usize = 256;

/// The packed kernel parameter blob. `#[repr(C)]` and fixed-size so it can
/// cross the FFI boundary by value into a `__grid_constant__` parameter.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FusedParams {
    raw: [u32; PARAM_BLOB_BYTES / 4],
}

// SAFETY: FusedParams is a plain-old-data fixed-size array of u32 with
// repr(C) — bitwise-copyable across the launch ABI exactly like the
// scalar parameters cudarc already marshals.
unsafe impl DeviceRepr for FusedParams {}

impl Default for FusedParams {
    fn default() -> Self {
        Self {
            raw: [0; PARAM_BLOB_BYTES / 4],
        }
    }
}

/// Host-side builder that packs stage fields into the blob and records
/// their offsets for the code generator.
pub struct ParamPacker {
    blob: FusedParams,
    words_used: usize,
    /// (macro_name, word_offset, c_type) per field.
    fields: Vec<(String, usize, &'static str)>,
}

impl ParamPacker {
    fn new() -> Self {
        Self {
            blob: FusedParams::default(),
            words_used: 0,
            fields: Vec::new(),
        }
    }

    fn push_word(
        &mut self,
        name: String,
        c_type: &'static str,
        word: u32,
    ) -> Result<(), FusionError> {
        if self.words_used >= PARAM_BLOB_BYTES / 4 {
            return Err(FusionError::ParamsTooLarge {
                cap: PARAM_BLOB_BYTES,
                need: (self.words_used + 1) * 4,
            });
        }
        self.blob.raw[self.words_used] = word;
        self.fields.push((name, self.words_used, c_type));
        self.words_used += 1;
        Ok(())
    }

    /// Add an `f32` field; returns the accessor macro name.
    pub fn f32(&mut self, stage: usize, field: &str, v: f32) -> Result<String, FusionError> {
        let name = format!("P{stage}_{field}");
        self.push_word(name.clone(), "float", v.to_bits())?;
        Ok(name)
    }

    /// Add a `u32` field; returns the accessor macro name.
    pub fn u32(&mut self, stage: usize, field: &str, v: u32) -> Result<String, FusionError> {
        let name = format!("P{stage}_{field}");
        self.push_word(name.clone(), "unsigned int", v)?;
        Ok(name)
    }

    /// Accessor macro definitions with baked constant-bank offsets.
    fn accessor_defines(&self) -> String {
        let mut s = String::new();
        for (name, word, c_type) in &self.fields {
            if *c_type == "float" {
                let _ = writeln!(s, "#define {name} (__uint_as_float(P.raw[{word}]))");
            } else {
                let _ = writeln!(s, "#define {name} (P.raw[{word}])");
            }
        }
        s
    }
}

/// One stage of a fused pipeline.
///
/// `build` packs kernel-friendly parameters into the blob (host side, once
/// per pipeline instantiation) and returns the device code: optional
/// `__device__` helper functions plus the exec snippet. The snippet reads
/// the stage input from the C variable named by `in_var` and must declare
/// and write its output as `float3 {out_var}`. Source stages ignore
/// `in_var`; sink stages write to the destination pointer and produce no
/// output variable.
pub trait FusedStage {
    /// Unique-ish name used in the pipeline cache key and kernel name.
    fn name(&self) -> String;
    /// Pack parameters; return (device helper fns, exec snippet).
    fn build(
        &self,
        stage_idx: usize,
        packer: &mut ParamPacker,
        in_var: &str,
        out_var: &str,
    ) -> Result<(String, String), FusionError>;
    /// Bytes this stage reads from the kernel's `src` argument (source
    /// stages). Lets the pipeline length-check `src` at launch.
    fn src_bytes_required(&self) -> Option<usize> {
        None
    }
    /// Elements this stage writes per image to `dst` (sink stages). Lets
    /// the pipeline length-check `dst` at launch.
    fn out_elems(&self, _dst_w: u32, _dst_h: u32) -> Option<usize> {
        None
    }
}

/// Role annotations for validation.
pub enum StageRole {
    /// Produces the initial value from `src` at the output coordinate.
    Source,
    /// Maps a value to a value.
    Map,
    /// Writes the final value to `dst`.
    Sink,
}

/// A composed pipeline: source + maps + sink over a destination grid.
pub struct FusedPipeline {
    kernel: Arc<CudaKernel>,
    params: FusedParams,
    dst_w: u32,
    dst_h: u32,
    source: String,
    /// Batch size (1 = unbatched kernel without the z prologue).
    batch: u32,
    /// Word index in the blob where the per-image src pointers start
    /// (batched pipelines only).
    ptr_base: usize,
    /// Per-image dst element stride (batched pipelines only).
    dst_stride: usize,
    /// Bytes the source stage reads from `src` (0 = unknown; no check).
    src_bytes: usize,
    /// Elements the sink stage writes per image (0 = unknown; no check).
    out_elems: usize,
}

type PipelineKernelCache = Mutex<HashMap<String, Arc<CudaKernel>>>;
static PIPELINE_KERNELS: OnceLock<PipelineKernelCache> = OnceLock::new();
const PIPELINE_CACHE_CAP: usize = 64;

impl FusedPipeline {
    /// Compose and compile (cached per generated-source hash).
    ///
    /// `stages` must be source, maps..., sink. The kernel is launched over
    /// the `dst_w x dst_h` output grid.
    pub fn build(
        ctx: &Arc<CudaContext>,
        stages: &[&dyn FusedStage],
        dst_w: u32,
        dst_h: u32,
    ) -> Result<Self, FusionError> {
        Self::build_inner(ctx, stages, dst_w, dst_h, 1, 0)
    }

    /// Compose a BATCHED pipeline: one launch processes `batch` images via
    /// `blockIdx.z` (the FKL DivergentBatchTransformDPP pattern, homogeneous
    /// variant — same chain per slice, per-image source pointers).
    ///
    /// Per-image src pointers are carried in the `__grid_constant__` blob
    /// and re-pointed in a prologue, so stage snippets are identical to the
    /// unbatched kernel. `dst` is one contiguous tensor; each image writes
    /// at `z * out_elems_per_image` (e.g. `3 * dst_w * dst_h` for the CHW
    /// sink → an NCHW batch tensor).
    pub fn build_batched(
        ctx: &Arc<CudaContext>,
        stages: &[&dyn FusedStage],
        dst_w: u32,
        dst_h: u32,
        batch: u32,
        out_elems_per_image: usize,
    ) -> Result<Self, FusionError> {
        if batch == 0 {
            return Err(FusionError::Pipeline("batch must be >= 1".into()));
        }
        Self::build_inner(ctx, stages, dst_w, dst_h, batch, out_elems_per_image)
    }

    fn build_inner(
        ctx: &Arc<CudaContext>,
        stages: &[&dyn FusedStage],
        dst_w: u32,
        dst_h: u32,
        batch: u32,
        out_elems_per_image: usize,
    ) -> Result<Self, FusionError> {
        if stages.len() < 2 {
            return Err(FusionError::Pipeline(
                "need at least a source and a sink stage".into(),
            ));
        }

        let mut packer = ParamPacker::new();
        let mut helpers = String::new();
        let mut body = String::new();
        let mut names = Vec::new();

        for (i, stage) in stages.iter().enumerate() {
            let in_var = if i == 0 {
                String::new()
            } else {
                format!("v{}", i - 1)
            };
            let out_var = if i + 1 == stages.len() {
                String::new()
            } else {
                format!("v{i}")
            };
            let (h, snippet) = stage.build(i, &mut packer, &in_var, &out_var)?;
            helpers.push_str(&h);
            let _ = writeln!(body, "    // stage {i}: {}", stage.name());
            body.push_str(&snippet);
            names.push(stage.name());
        }

        // Batched: reserve blob slots for the per-image src pointers
        // (lo/hi u32 pairs, filled at launch time) and emit the z prologue
        // that re-points src/dst — stage snippets stay untouched.
        let mut prologue = String::new();
        let mut ptr_base = 0usize;
        if batch > 1 {
            ptr_base = packer.words_used;
            for b in 0..batch {
                packer.u32(usize::MAX, &format!("srclo{b}"), 0)?;
                packer.u32(usize::MAX, &format!("srchi{b}"), 0)?;
            }
            // Pointer slots are accessed by dynamic index, not macros —
            // drop their auto-generated accessors.
            packer
                .fields
                .truncate(packer.fields.len() - 2 * batch as usize);
            let _ = write!(
                prologue,
                "    unsigned int b = blockIdx.z;\n\
                 \x20   src = (const unsigned char*)(((unsigned long long)P.raw[{base}u + 2u * b + 1u] << 32) \
                 | (unsigned long long)P.raw[{base}u + 2u * b]);\n\
                 \x20   dst += (size_t)b * {stride}u;\n",
                base = ptr_base,
                stride = out_elems_per_image,
            );
        }

        let accessors = packer.accessor_defines();
        let kernel_name = "fused_pipeline";
        let source = format!(
            r#"
struct FusedParams {{ unsigned int raw[{words}]; }};

{accessors}
{helpers}
extern "C" __global__ void {kernel_name}(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    __grid_constant__ const FusedParams P,
    unsigned int dst_w,
    unsigned int dst_h
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

{prologue}{body}}}
"#,
            words = PARAM_BLOB_BYTES / 4,
        );

        // Cache by full source (shape + baked constants that live in code;
        // blob values do NOT enter the source, so same-shape pipelines with
        // different parameters share one kernel).
        let key = format!("{}|{source}", names.join("+"));
        let cache = PIPELINE_KERNELS.get_or_init(Default::default);
        let cached = cache
            .lock()
            .expect("fusion cache poisoned")
            .get(&key)
            .cloned();
        let kernel = if let Some(hit) = cached {
            hit
        } else {
            let built = Arc::new(
                try_compile_with_l1(ctx, &source, kernel_name).map_err(FusionError::Cuda)?,
            );
            let mut map = cache.lock().expect("fusion cache poisoned");
            if map.len() >= PIPELINE_CACHE_CAP {
                // Evict one entry, not the whole map — wholesale clearing
                // stampedes >cap shape sets into ~1s NVRTC recompiles each.
                if let Some(k) = map.keys().next().cloned() {
                    map.remove(&k);
                }
            }
            map.entry(key).or_insert(built).clone()
        };

        // Size contracts from the stages, so the launch paths can length-
        // check the buffers the kernel will actually touch.
        let src_bytes = stages
            .iter()
            .find_map(|st| st.src_bytes_required())
            .unwrap_or(0);
        let out_elems = stages
            .iter()
            .find_map(|st| st.out_elems(dst_w, dst_h))
            .unwrap_or(0);
        if batch > 1 && out_elems > 0 && out_elems_per_image < out_elems {
            return Err(FusionError::Pipeline(format!(
                "out_elems_per_image {out_elems_per_image} is smaller than the sink's \
                 per-image output ({out_elems} elements)"
            )));
        }

        Ok(Self {
            kernel,
            params: packer.blob,
            dst_w,
            dst_h,
            source,
            batch,
            ptr_base,
            dst_stride: out_elems_per_image,
            src_bytes,
            out_elems,
        })
    }

    /// The generated CUDA source (introspection / debugging).
    pub fn generated_source(&self) -> &str {
        &self.source
    }

    /// Launch a batched pipeline: `srcs.len()` must equal the built batch
    /// size, `dst` holds `batch * out_elems_per_image` elements. All
    /// sources must live on `stream`'s device; the blob carries their raw
    /// pointers, so the borrows here keep them alive through the enqueue.
    pub fn launch_batched(
        &self,
        stream: &Arc<CudaStream>,
        srcs: &[&CudaSlice<u8>],
        dst: &mut CudaSlice<f32>,
    ) -> Result<(), FusionError> {
        use cudarc::driver::DevicePtr;
        if self.batch as usize != srcs.len() {
            return Err(FusionError::Pipeline(format!(
                "pipeline built for batch {}, got {} sources",
                self.batch,
                srcs.len()
            )));
        }
        if dst.len() < self.batch as usize * self.dst_stride {
            return Err(FusionError::Pipeline(format!(
                "dst holds {} elements; batch needs {}",
                dst.len(),
                self.batch as usize * self.dst_stride
            )));
        }
        let launch_ord = stream.context().ordinal();
        for (b, s) in srcs.iter().enumerate() {
            if s.stream().context().ordinal() != launch_ord {
                return Err(FusionError::Pipeline(format!(
                    "source {b} lives on device {}, launch stream is device {launch_ord}",
                    s.stream().context().ordinal()
                )));
            }
            if self.src_bytes > 0 && s.len() < self.src_bytes {
                return Err(FusionError::Pipeline(format!(
                    "source {b} holds {} bytes; the source stage reads {}",
                    s.len(),
                    self.src_bytes
                )));
            }
        }
        let mut params = self.params;
        let mut guards = Vec::with_capacity(srcs.len());
        for (b, s) in srcs.iter().enumerate() {
            let (ptr, guard) = s.device_ptr(stream);
            params.raw[self.ptr_base + 2 * b] = (ptr & 0xffff_ffff) as u32;
            params.raw[self.ptr_base + 2 * b + 1] = (ptr >> 32) as u32;
            guards.push(guard);
        }
        let cfg = cudarc::driver::LaunchConfig {
            block_dim: (32, 8, 1),
            grid_dim: (self.dst_w.div_ceil(32), self.dst_h.div_ceil(8), self.batch),
            shared_mem_bytes: 0,
        };
        self.kernel
            .launch_builder(stream)
            .arg(srcs[0])
            .arg(dst)
            .arg(&params)
            .arg(&self.dst_w)
            .arg(&self.dst_h)
            .launch_cfg(cfg)
            .map_err(|e| FusionError::Cuda(e.to_string()))
    }

    /// Launch over the destination grid. `src` and `dst` element meanings
    /// are defined by the pipeline's source/sink stages.
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<f32>,
    ) -> Result<(), FusionError> {
        if self.batch > 1 {
            return Err(FusionError::Pipeline(
                "pipeline was built batched; use launch_batched".into(),
            ));
        }
        if self.src_bytes > 0 && src.len() < self.src_bytes {
            return Err(FusionError::Pipeline(format!(
                "src holds {} bytes; the source stage reads {}",
                src.len(),
                self.src_bytes
            )));
        }
        if self.out_elems > 0 && dst.len() < self.out_elems {
            return Err(FusionError::Pipeline(format!(
                "dst holds {} elements; the sink writes {}",
                dst.len(),
                self.out_elems
            )));
        }
        self.kernel
            .launch_builder(stream)
            .arg(src)
            .arg(dst)
            .arg(&self.params)
            .arg(&self.dst_w)
            .arg(&self.dst_h)
            .launch_2d(
                self.dst_w,
                self.dst_h,
                make_config(self.dst_w, self.dst_h, None),
            )
            .map_err(|e| FusionError::Cuda(e.to_string()))
    }
}

// ── Stage library v1 ──────────────────────────────────────────────────────────

/// Source: bilinear-resample an interleaved u8 RGB image on the half-pixel
/// grid, producing an f32 RGB value in [0, 255] — the same sampling the
/// hand-fused preprocessor uses.
pub struct ReadU8RgbBilinear {
    /// Source image width.
    pub src_w: u32,
    /// Source image height.
    pub src_h: u32,
    /// Destination width (for the host-precomputed half-pixel coefficients).
    pub dst_w: u32,
    /// Destination height.
    pub dst_h: u32,
}

impl FusedStage for ReadU8RgbBilinear {
    fn name(&self) -> String {
        "read_u8rgb_bilinear".into()
    }
    fn build(
        &self,
        i: usize,
        p: &mut ParamPacker,
        _in: &str,
        out: &str,
    ) -> Result<(String, String), FusionError> {
        let sw = p.u32(i, "sw", self.src_w)?;
        let sh = p.u32(i, "sh", self.src_h)?;
        // Half-pixel mapping `s = a*d + b`, coefficients built on the host
        // (the FKL build()/exec() split: no per-pixel divides on device).
        let axv = self.src_w as f32 / self.dst_w as f32;
        let ayv = self.src_h as f32 / self.dst_h as f32;
        let ax = p.f32(i, "ax", axv)?;
        let bx = p.f32(i, "bx", 0.5 * axv - 0.5)?;
        let ay = p.f32(i, "ay", ayv)?;
        let by = p.f32(i, "by", 0.5 * ayv - 0.5)?;
        let snippet = format!(
            r#"    float sxf = fmaxf({ax} * (float)x + {bx}, 0.0f);
    float syf = fmaxf({ay} * (float)y + {by}, 0.0f);
    unsigned int sx0 = min((unsigned int)sxf, {sw} - 1u);
    unsigned int sy0 = min((unsigned int)syf, {sh} - 1u);
    unsigned int sx1 = min(sx0 + 1u, {sw} - 1u);
    unsigned int sy1 = min(sy0 + 1u, {sh} - 1u);
    float wx = sxf - (float)sx0;
    float wy = syf - (float)sy0;
    size_t r0 = (size_t)sy0 * {sw} * 3u;
    size_t r1 = (size_t)sy1 * {sw} * 3u;
    float3 {out};
    {{
        float w00 = (1.0f - wy) * (1.0f - wx);
        float w01 = (1.0f - wy) * wx;
        float w10 = wy * (1.0f - wx);
        float w11 = wy * wx;
        {out}.x = w00 * (float)__ldg(&src[r0 + (size_t)sx0 * 3u + 0u])
                + w01 * (float)__ldg(&src[r0 + (size_t)sx1 * 3u + 0u])
                + w10 * (float)__ldg(&src[r1 + (size_t)sx0 * 3u + 0u])
                + w11 * (float)__ldg(&src[r1 + (size_t)sx1 * 3u + 0u]);
        {out}.y = w00 * (float)__ldg(&src[r0 + (size_t)sx0 * 3u + 1u])
                + w01 * (float)__ldg(&src[r0 + (size_t)sx1 * 3u + 1u])
                + w10 * (float)__ldg(&src[r1 + (size_t)sx0 * 3u + 1u])
                + w11 * (float)__ldg(&src[r1 + (size_t)sx1 * 3u + 1u]);
        {out}.z = w00 * (float)__ldg(&src[r0 + (size_t)sx0 * 3u + 2u])
                + w01 * (float)__ldg(&src[r0 + (size_t)sx1 * 3u + 2u])
                + w10 * (float)__ldg(&src[r1 + (size_t)sx0 * 3u + 2u])
                + w11 * (float)__ldg(&src[r1 + (size_t)sx1 * 3u + 2u]);
    }}
"#
        );
        Ok((String::new(), snippet))
    }
    fn src_bytes_required(&self) -> Option<usize> {
        Some(self.src_w as usize * self.src_h as usize * 3)
    }
}

/// Map: per-channel `v * scale + bias` (e.g. `(x/255 - mean)/std` folded).
pub struct Normalize {
    /// Per-channel multiplicative factors.
    pub scale: [f32; 3],
    /// Per-channel additive biases.
    pub bias: [f32; 3],
}

impl FusedStage for Normalize {
    fn name(&self) -> String {
        "normalize".into()
    }
    fn build(
        &self,
        i: usize,
        p: &mut ParamPacker,
        inv: &str,
        out: &str,
    ) -> Result<(String, String), FusionError> {
        let s0 = p.f32(i, "s0", self.scale[0])?;
        let s1 = p.f32(i, "s1", self.scale[1])?;
        let s2 = p.f32(i, "s2", self.scale[2])?;
        let b0 = p.f32(i, "b0", self.bias[0])?;
        let b1 = p.f32(i, "b1", self.bias[1])?;
        let b2 = p.f32(i, "b2", self.bias[2])?;
        let snippet = format!(
            "    float3 {out};\n    {out}.x = {inv}.x * {s0} + {b0};\n    {out}.y = {inv}.y * {s1} + {b1};\n    {out}.z = {inv}.z * {s2} + {b2};\n"
        );
        Ok((String::new(), snippet))
    }
}

/// Map: RGB → gray (ITU-R BT.601 f32 weights), replicated to all lanes.
pub struct RgbToGray;

impl FusedStage for RgbToGray {
    fn name(&self) -> String {
        "rgb_to_gray".into()
    }
    fn build(
        &self,
        _i: usize,
        _p: &mut ParamPacker,
        inv: &str,
        out: &str,
    ) -> Result<(String, String), FusionError> {
        let snippet = format!(
            "    float3 {out};\n    {out}.x = 0.299f * {inv}.x + 0.587f * {inv}.y + 0.114f * {inv}.z;\n    {out}.y = {out}.x;\n    {out}.z = {out}.x;\n"
        );
        Ok((String::new(), snippet))
    }
}

/// Sink: write planar CHW f32 (3 planes).
pub struct WriteChwF32;

impl FusedStage for WriteChwF32 {
    fn name(&self) -> String {
        "write_chw_f32".into()
    }
    fn build(
        &self,
        _i: usize,
        _p: &mut ParamPacker,
        inv: &str,
        _out: &str,
    ) -> Result<(String, String), FusionError> {
        let snippet = format!(
            "    size_t plane = (size_t)dst_w * dst_h;\n    size_t di = (size_t)y * dst_w + x;\n    dst[di] = {inv}.x;\n    dst[di + plane] = {inv}.y;\n    dst[di + 2u * plane] = {inv}.z;\n"
        );
        Ok((String::new(), snippet))
    }
    fn out_elems(&self, dst_w: u32, dst_h: u32) -> Option<usize> {
        Some(3 * dst_w as usize * dst_h as usize)
    }
}

/// Sink: write a single-channel f32 plane from lane x.
pub struct WriteC1F32;

impl FusedStage for WriteC1F32 {
    fn name(&self) -> String {
        "write_c1_f32".into()
    }
    fn build(
        &self,
        _i: usize,
        _p: &mut ParamPacker,
        inv: &str,
        _out: &str,
    ) -> Result<(String, String), FusionError> {
        let snippet = format!("    dst[(size_t)y * dst_w + x] = {inv}.x;\n");
        Ok((String::new(), snippet))
    }
    fn out_elems(&self, dst_w: u32, dst_h: u32) -> Option<usize> {
        Some(dst_w as usize * dst_h as usize)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::pattern_u8;

    fn ctx_stream() -> (Arc<CudaContext>, Arc<CudaStream>) {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        (ctx, stream)
    }

    /// CPU reference of the fused semantics (f32 register flow, half-pixel
    /// bilinear), for tolerance comparison.
    #[allow(clippy::too_many_arguments)] // test oracle mirrors the kernel's parameter surface
    fn cpu_reference(
        src: &[u8],
        sw: usize,
        sh: usize,
        dw: usize,
        dh: usize,
        scale: [f32; 3],
        bias: [f32; 3],
        gray: bool,
    ) -> Vec<f32> {
        let ax = sw as f32 / dw as f32;
        let ay = sh as f32 / dh as f32;
        let (bx, by) = (0.5 * ax - 0.5, 0.5 * ay - 0.5);
        let planes = if gray { 1 } else { 3 };
        let mut out = vec![0f32; planes * dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let sxf = (ax * x as f32 + bx).max(0.0);
                let syf = (ay * y as f32 + by).max(0.0);
                let sx0 = (sxf as usize).min(sw - 1);
                let sy0 = (syf as usize).min(sh - 1);
                let sx1 = (sx0 + 1).min(sw - 1);
                let sy1 = (sy0 + 1).min(sh - 1);
                let wx = sxf - sx0 as f32;
                let wy = syf - sy0 as f32;
                let mut v = [0f32; 3];
                for (c, vc) in v.iter_mut().enumerate() {
                    let p = |yy: usize, xx: usize| src[(yy * sw + xx) * 3 + c] as f32;
                    *vc = (1.0 - wy) * (1.0 - wx) * p(sy0, sx0)
                        + (1.0 - wy) * wx * p(sy0, sx1)
                        + wy * (1.0 - wx) * p(sy1, sx0)
                        + wy * wx * p(sy1, sx1);
                }
                if gray {
                    let g0 = 0.299 * (v[0] * scale[0] + bias[0])
                        + 0.587 * (v[1] * scale[1] + bias[1])
                        + 0.114 * (v[2] * scale[2] + bias[2]);
                    out[y * dw + x] = g0;
                } else {
                    for c in 0..3 {
                        out[c * dw * dh + y * dw + x] = v[c] * scale[c] + bias[c];
                    }
                }
            }
        }
        out
    }

    /// Composed resize→normalize→CHW matches the CPU reference of the same
    /// f32 semantics within float tolerance.
    #[test]
    fn fused_preprocess_matches_reference() {
        let (ctx, stream) = ctx_stream();
        let (sw, sh, dw, dh) = (129usize, 97usize, 64usize, 48usize);
        let host = pattern_u8(sw * sh * 3);
        let scale = [
            1.0 / 255.0 / 0.229,
            1.0 / 255.0 / 0.224,
            1.0 / 255.0 / 0.225,
        ];
        let bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

        let read = ReadU8RgbBilinear {
            src_w: sw as u32,
            src_h: sh as u32,
            dst_w: dw as u32,
            dst_h: dh as u32,
        };
        let norm = Normalize { scale, bias };
        let sink = WriteChwF32;
        let pipe =
            FusedPipeline::build(&ctx, &[&read, &norm, &sink], dw as u32, dh as u32).unwrap();

        let d_src = stream.clone_htod(&host).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(3 * dw * dh).unwrap();
        pipe.launch(&stream, &d_src, &mut d_dst).unwrap();
        let got = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        let want = cpu_reference(&host, sw, sh, dw, dh, scale, bias, false);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "element {i}: gpu {g} vs cpu {w}"
            );
        }
    }

    /// A NOVEL composition not hand-written anywhere: resize → normalize →
    /// gray → single-plane write. The engine generates it from the same
    /// stage library.
    #[test]
    fn fused_novel_gray_chain() {
        let (ctx, stream) = ctx_stream();
        let (sw, sh, dw, dh) = (100usize, 80usize, 47usize, 33usize);
        let host = pattern_u8(sw * sh * 3);
        let scale = [1.0 / 255.0; 3];
        let bias = [0.0; 3];

        let read = ReadU8RgbBilinear {
            src_w: sw as u32,
            src_h: sh as u32,
            dst_w: dw as u32,
            dst_h: dh as u32,
        };
        let norm = Normalize { scale, bias };
        let gray = RgbToGray;
        let sink = WriteC1F32;
        let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &gray, &sink], dw as u32, dh as u32)
            .unwrap();

        let d_src = stream.clone_htod(&host).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(dw * dh).unwrap();
        pipe.launch(&stream, &d_src, &mut d_dst).unwrap();
        let got = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        let want = cpu_reference(&host, sw, sh, dw, dh, scale, bias, true);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "element {i}: gpu {g} vs cpu {w}"
            );
        }
    }

    /// Same-shape pipelines with different parameter VALUES share one
    /// compiled kernel (the blob is data, not code).
    #[test]
    fn same_shape_pipelines_share_kernel() {
        let (ctx, _stream) = ctx_stream();
        let mk = |mean: f32| {
            let read = ReadU8RgbBilinear {
                src_w: 64,
                src_h: 64,
                dst_w: 32,
                dst_h: 32,
            };
            let norm = Normalize {
                scale: [1.0; 3],
                bias: [mean; 3],
            };
            FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], 32, 32).unwrap()
        };
        let a = mk(0.0);
        let b = mk(9.5);
        assert!(Arc::ptr_eq(&a.kernel, &b.kernel), "kernel must be shared");
        assert_ne!(a.params.raw, b.params.raw, "params must differ");
    }

    /// Throughput probe (ignored): fused engine vs the hand-written fused
    /// preprocess kernel at 1080p -> 640.
    #[test]
    #[ignore]
    fn probe_fused_1080p() {
        let (ctx, stream) = ctx_stream();
        let (sw, sh, dw, dh) = (1920usize, 1080usize, 640usize, 640usize);
        let host = pattern_u8(sw * sh * 3);
        let read = ReadU8RgbBilinear {
            src_w: sw as u32,
            src_h: sh as u32,
            dst_w: dw as u32,
            dst_h: dh as u32,
        };
        let norm = Normalize {
            scale: [1.0 / 255.0; 3],
            bias: [0.0; 3],
        };
        let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], dw as u32, dh as u32)
            .unwrap();
        let d_src = stream.clone_htod(&host).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(3 * dw * dh).unwrap();

        for _ in 0..100 {
            pipe.launch(&stream, &d_src, &mut d_dst).unwrap();
        }
        stream.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                pipe.launch(&stream, &d_src, &mut d_dst).unwrap();
            }
            stream.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / 200.0);
        }
        println!("fused engine 1080p->640 CHW norm: {best:.3} ms/op (min of 5)");
    }
}

#[cfg(test)]
mod show_source {
    use super::*;

    /// Print the generated kernel for the Pipeline-A chain (run with
    /// `-- --ignored --nocapture` to inspect).
    #[test]
    #[ignore]
    fn print_pipeline_a_source() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let read = ReadU8RgbBilinear {
            src_w: 1920,
            src_h: 1080,
            dst_w: 640,
            dst_h: 640,
        };
        let norm = Normalize {
            scale: [
                1.0 / 255.0 / 0.229,
                1.0 / 255.0 / 0.224,
                1.0 / 255.0 / 0.225,
            ],
            bias: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        };
        let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], 640, 640).unwrap();
        println!("{}", pipe.generated_source());
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::cuda::color::test_utils::pattern_u8;

    /// A batch-4 launch must produce exactly the concatenation of four
    /// single-image launches of the same pipeline shape.
    #[test]
    fn batched_matches_single_launches() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        let (sw, sh, dw, dh) = (129u32, 97u32, 64u32, 48u32);
        let out_elems = 3 * (dw as usize) * (dh as usize);

        let read = ReadU8RgbBilinear {
            src_w: sw,
            src_h: sh,
            dst_w: dw,
            dst_h: dh,
        };
        let norm = Normalize {
            scale: [1.0 / 255.0; 3],
            bias: [-0.5; 3],
        };

        let single = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], dw, dh).unwrap();
        let batched =
            FusedPipeline::build_batched(&ctx, &[&read, &norm, &WriteChwF32], dw, dh, 4, out_elems)
                .unwrap();

        // Four distinct images.
        let hosts: Vec<Vec<u8>> = (0..4)
            .map(|i| {
                let mut v = pattern_u8((sw * sh * 3) as usize);
                v.iter_mut().for_each(|b| *b = b.wrapping_add(i * 37));
                v
            })
            .collect();
        let d_srcs: Vec<CudaSlice<u8>> = hosts
            .iter()
            .map(|h| stream.clone_htod(h).unwrap())
            .collect();

        // Reference: single launches.
        let mut want = Vec::new();
        for d in &d_srcs {
            let mut d_dst = stream.alloc_zeros::<f32>(out_elems).unwrap();
            single.launch(&stream, d, &mut d_dst).unwrap();
            want.extend(stream.clone_dtoh(&d_dst).unwrap());
        }

        // Batched launch into one NCHW tensor.
        let mut d_batch = stream.alloc_zeros::<f32>(4 * out_elems).unwrap();
        let refs: Vec<&CudaSlice<u8>> = d_srcs.iter().collect();
        batched
            .launch_batched(&stream, &refs, &mut d_batch)
            .unwrap();
        let got = stream.clone_dtoh(&d_batch).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).to_bits() == 0 || (g - w).abs() == 0.0,
                "element {i}: batched {g} vs single {w}"
            );
        }
    }

    /// Undersized buffers are typed errors, not OOB device accesses: the
    /// stages publish their size contracts and both launch paths check.
    #[test]
    fn undersized_buffers_error() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        let read = ReadU8RgbBilinear {
            src_w: 64,
            src_h: 64,
            dst_w: 32,
            dst_h: 32,
        };
        let norm = Normalize {
            scale: [1.0; 3],
            bias: [0.0; 3],
        };
        let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], 32, 32).unwrap();
        let short_src = stream.alloc_zeros::<u8>(64 * 64 * 3 - 1).unwrap();
        let mut dst = stream.alloc_zeros::<f32>(3 * 32 * 32).unwrap();
        assert!(pipe.launch(&stream, &short_src, &mut dst).is_err());
        let src = stream.alloc_zeros::<u8>(64 * 64 * 3).unwrap();
        let mut short_dst = stream.alloc_zeros::<f32>(3 * 32 * 32 - 1).unwrap();
        assert!(pipe.launch(&stream, &src, &mut short_dst).is_err());
        // Batched: undersized per-image source is caught too.
        let batched = FusedPipeline::build_batched(
            &ctx,
            &[&read, &norm, &WriteChwF32],
            32,
            32,
            2,
            3 * 32 * 32,
        )
        .unwrap();
        let mut bdst = stream.alloc_zeros::<f32>(2 * 3 * 32 * 32).unwrap();
        assert!(batched
            .launch_batched(&stream, &[&src, &short_src], &mut bdst)
            .is_err());
    }

    /// Wrong source count and unbatched-launch-on-batched both error.
    #[test]
    fn batch_misuse_errors() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        let read = ReadU8RgbBilinear {
            src_w: 64,
            src_h: 64,
            dst_w: 32,
            dst_h: 32,
        };
        let norm = Normalize {
            scale: [1.0; 3],
            bias: [0.0; 3],
        };
        let out_elems = 3 * 32 * 32;
        let batched =
            FusedPipeline::build_batched(&ctx, &[&read, &norm, &WriteChwF32], 32, 32, 2, out_elems)
                .unwrap();
        let host = pattern_u8(64 * 64 * 3);
        let d = stream.clone_htod(&host).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(2 * out_elems).unwrap();
        assert!(batched.launch_batched(&stream, &[&d], &mut d_dst).is_err());
        assert!(batched.launch(&stream, &d, &mut d_dst).is_err());
    }

    /// Batch throughput probe (ignored): batch-4 1080p -> 640 CHW norm.
    #[test]
    #[ignore]
    fn probe_batch4_1080p() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        let (sw, sh, dw, dh) = (1920u32, 1080u32, 640u32, 640u32);
        let out_elems = 3 * (dw as usize) * (dh as usize);
        let read = ReadU8RgbBilinear {
            src_w: sw,
            src_h: sh,
            dst_w: dw,
            dst_h: dh,
        };
        let norm = Normalize {
            scale: [1.0 / 255.0; 3],
            bias: [0.0; 3],
        };
        let pipe =
            FusedPipeline::build_batched(&ctx, &[&read, &norm, &WriteChwF32], dw, dh, 4, out_elems)
                .unwrap();
        let host = pattern_u8((sw * sh * 3) as usize);
        let d_srcs: Vec<CudaSlice<u8>> =
            (0..4).map(|_| stream.clone_htod(&host).unwrap()).collect();
        let refs: Vec<&CudaSlice<u8>> = d_srcs.iter().collect();
        let mut d_dst = stream.alloc_zeros::<f32>(4 * out_elems).unwrap();

        for _ in 0..50 {
            pipe.launch_batched(&stream, &refs, &mut d_dst).unwrap();
        }
        stream.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..100 {
                pipe.launch_batched(&stream, &refs, &mut d_dst).unwrap();
            }
            stream.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / 100.0);
        }
        println!(
            "fused batch-4 1080p->640 CHW norm: {best:.3} ms/batch ({:.3} ms/image, min of 5)",
            best / 4.0
        );
    }
}
