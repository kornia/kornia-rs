# CUDA kernel fusion (FKL-style)

Compose per-op stages into **one generated CUDA kernel** where
intermediates flow through registers — no global-memory round trips
between ops. The design follows the [Fused Kernel Library]
(https://github.com/Libraries-Openly-Fused) build/exec model, but composes
at **runtime** via NVRTC (compiled once per pipeline shape, then cached;
measured at exact performance parity with FKL's compile-time templates on
Jetson AGX Orin).

```bash
cargo run -p cuda-fusion --release
```

The example composes:

1. the DNN-preprocess chain `resize → normalize(ImageNet) → planar CHW`
   (≈0.13 ms/frame for 1080p → 640×640 on Orin — one kernel, one u8 read,
   one f32 write);
2. a **novel** chain `resize → normalize → gray → single-plane write` that
   exists nowhere as hand-written kernel code;
3. prints the generated CUDA source so you can see exactly what runs.

## Composing a pipeline

```rust
use kornia_imgproc::cuda::fusion::*;

let read = ReadU8RgbBilinear { src_w, src_h, dst_w, dst_h }; // Source
let norm = Normalize { scale, bias };                        // Map
let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], dst_w, dst_h)?;
pipe.launch(&stream, &d_src, &mut d_dst)?;                   // one kernel
```

Stages must be `Source, Map..., Sink`. Batched variants
(`build_batched`/`launch_batched`) add a `blockIdx.z` prologue so one
launch processes N images (bit-identical to N single launches).

## Writing your own stage

Implement `FusedStage`: pack parameters into the blob and emit a CUDA
snippet that transforms the register value `float3 v`:

```rust
struct Saturate { gain: f32 }

impl FusedStage for Saturate {
    fn name(&self) -> String { "saturate".into() }
    fn build(&self, idx: usize, p: &mut ParamPacker, i: &str, o: &str)
        -> Result<(String, String), FusionError>
    {
        let g = p.f32(idx, "gain", self.gain)?; // -> constant-bank macro
        Ok((String::new(), format!(
            "float3 {o}; float m = ({i}.x + {i}.y + {i}.z) * (1.0f/3.0f);\n\
             {o}.x = m + ({i}.x - m) * {g};\n\
             {o}.y = m + ({i}.y - m) * {g};\n\
             {o}.z = m + ({i}.z - m) * {g};\n"
        )))
    }
}
```

Parameters live in a single `__grid_constant__` POD blob (constant-bank
reads: zero registers, warp-broadcast), so chains don't pay register cost
for carrying state. Same-shape pipelines with different parameter values
share one compiled kernel — params are data, shapes are code.

## Precision contract

Fused chains keep intermediates in f32 registers, so output is NOT
byte-equal to running the u8 ops separately through buffers — it is a
documented higher-precision contract (no intermediate quantization), the
same one the hand-fused `Preprocessor` uses.
