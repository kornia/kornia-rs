# CUDA kernel fusion

Compose per-op stages into **one generated CUDA kernel** where
intermediates flow through registers — no global-memory round trips
between ops. The build/exec stage split and the `__grid_constant__`
parameter-blob technique are borrowed from the [Fused Kernel Library]
(https://github.com/Libraries-Openly-Fused) (Amoros / Nuñez / Peña);
composition here happens at **runtime** via NVRTC (compiled once per
pipeline shape, then cached).

**Scope — this is NOT a general FKL equivalent.** The current engine
composes *linear, per-output-pixel transform chains only*: one `Source`
stage (single input buffer) → `Map` stages passing a `float3` value in
registers → one `Sink` stage (single output buffer). No multi-input
stages, no reductions, no other inter-stage value types yet. FKL's
template machinery composes considerably more (arbitrary value types,
multi-operand operations, reduction patterns). Within this transform-chain
scope, per-stage code is arbitrary CUDA.

```bash
cargo run -p cuda-fusion --release
```

The example composes:

1. the DNN-preprocess chain `resize → normalize(ImageNet) → planar CHW`
   (≈0.13 ms/frame for 1080p → 640×640 on Jetson AGX Orin — one kernel,
   one u8 read, one f32 write);
2. a chain with no hand-written kernel anywhere in the library
   (`resize → normalize → gray → single-plane write`), generated from the
   same stage library;
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

## Benchmark vs the FKL binary — one pipeline only

For the one pipeline both implementations express (bilinear resize →
Mul(1/255) → planar split; FKL's `TransformDPP` example), measured
same-minute against the FKL C++ binary compiled on the same Jetson AGX
Orin (sm_87, locked clocks):

| 1080p → 640×640 | this engine | FKL |
|---|---|---|
| single | 0.13–0.14 ms | 0.12–0.13 ms |
| batch-4 | 0.49–0.50 ms | 0.49 ms |

This says the *generated kernel* for this chain is as good as FKL's
template-instantiated one — it says nothing about the many pipeline
shapes FKL can express and this engine cannot (see Scope above).

## Roadmap

The engine is planned to grow toward general pipeline composition in the
spirit of FKL and GStreamer:

- typed inter-stage values (`float`, `float3`, `float4`, packed u8) with
  build-time compatibility checking;
- multi-input stages (two-image ops: add_weighted, masking) via the same
  per-image pointer blob the batch path already uses;
- reduction sinks (histograms, statistics) as a second grid contract;
- longer term: branching pipeline graphs (GStreamer-style tees/muxes)
  and a Python surface for runtime-composed chains.

## Precision contract

Fused chains keep intermediates in f32 registers, so output is NOT
byte-equal to running the u8 ops separately through buffers — it is a
documented higher-precision contract (no intermediate quantization), the
same one the hand-fused `Preprocessor` uses.
