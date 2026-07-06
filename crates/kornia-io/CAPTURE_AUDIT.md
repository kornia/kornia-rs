# Video capture audit — status & follow-ups

Audit and hardening of the V4L2 and GStreamer capture backends in `kornia-io`,
plus a zero-copy V4L Python binding in `kornia-py`, a generic `GstFrame`, and a
**proven** (prototype) GStreamer→CUDA→PyTorch zero-copy path on Jetson.

This document tracks what landed in this PR and everything still missing.

---

## 1. Delivered in this PR

### V4L2 (`crates/kornia-io/src/v4l/`)
- **Sound zero-copy buffer recycling** (`stream.rs`): a dequeued buffer is only
  re-queued to the kernel once no `MmapBuffer` clone is still alive
  (`Arc::strong_count`-gated). Prevents the kernel from overwriting memory a caller
  is still reading (data-race UB). Returns `BuffersExhausted` if every buffer is
  pinned, instead of racing.
- **Negotiated-size correctness** (`mod.rs`): reads back `device.format()`, stores
  the actual size, exposes `size()`, warns on clamp.
- `grab_frame` surfaces real device errors instead of swallowing them to `Ok(None)`
  (timeout → `None`, exhaustion → typed error).
- `set_timeout`, explicit `start_streaming` (no discarded first frame),
  `eprintln!` → `log`.

### GStreamer (`crates/kornia-io/src/gstreamer/`)
- **Bus-error surfacing** (`capture.rs`): background bus-watch thread records fatal
  errors; grab methods return `PipelineError` instead of looping on `None`. Adds
  `last_error()` / `is_running()`.
- **Generic `GstFrame`** (`frame.rs`, new): `VideoInfo`-driven, zero-copy frame
  holding the mapped buffer keepalive + format + strides + PTS/duration.
  - Generic element type + channels: `to_image_u8::<C>()` / `to_image_u16::<C>()`.
  - Format table (`RGB/BGR/RGBA/GRAY8` → u8, `GRAY16_LE` → u16); mismatched
    type/channels are **rejected** (no silent mislabel).
  - **Stride-correct**: zero-copy for tightly-packed rows, packed copy for padded
    rows (fixes latent bug for non-4-aligned widths).
- `grab()` → `Option<GstFrame>`; `grab_rgb8` / `grab_mono8` / `grab_mono16`
  convenience wrappers (now stride-correct + format-validated).
- `VideoReader::grab_mono8` + format-guarded `grab_rgb8`; `VideoWriter::write_owned`
  (zero-copy write).

### Python (`kornia-py/src/io/v4l.rs`, new)
- `kornia_rs.capture.V4lCapture` + `V4lFrame`:
  - **Zero-copy**: `grab()` moves the `MmapBuffer` (no copy); `frame.raw` is a
    read-only `memoryview` (PEP-3118); `np.frombuffer(frame)` is a view (`owndata=False`).
  - `frame.image` → lazily decoded **`kornia_rs.image.Image`** (cached).
  - Metadata: `.timestamp`, `.sequence`, `.pixel_format`, `.is_encoded`, `.size()`.
  - GIL released during blocking grab (`py.detach`); context-manager support.
  - Decode lives in `kornia-py` (which already has `kornia-imgproc`), so
    **`kornia-io` has no imgproc/`wide` dependency**.

### Verified
- `cargo test -p kornia-io --features v4l,gstreamer` — 64 tests pass (incl. new
  V4L borrow-tracking, GStreamer mono16 + padded-stride + bus-error tests) against
  GStreamer 1.18.
- Live webcam (V4L): YUYV + MJPG, zero-copy `memoryview`, buffer-pinning guard.
- Affected examples build (`v4l`, `rtspcam`, `video_player`, `video_write`).
- **GStreamer→CUDA→PyTorch zero-copy proven on Jetson Orin** (JP6 / CUDA 12.6) —
  see §5.

---

## 2. Missing — V4L Python binding

- [ ] Camera controls exposed (`set_control`: brightness/contrast/exposure/…).
      Rust `V4lVideoCapture::set_control` exists; not wired to Python.
- [ ] `.pyi` type stubs for `V4lCapture` / `V4lFrame` (rest of `kornia-py` ships stubs).
- [ ] Dedicated `CaptureError` exception type (currently `RuntimeError`).
- [ ] UYVY decode (only YUYV/MJPG today; UYVY errors).
- [ ] In-suite tests (camera-gated; mark `@pytest.mark.skipif(no /dev/video*)`).
- [ ] Docs / example notebook.

## 3. Missing — GStreamer Python binding (not started)

- [ ] `kornia_rs.capture.StreamCapture` (arbitrary pipeline): `start`,
      `grab`/`grab_rgb8`/`grab_mono8`/`grab_mono16`, `get_fps`, `last_error`,
      `is_running`, `close`, context manager.
- [ ] Expose `GstFrame` in Python mirroring `V4lFrame`: `.raw` (buffer protocol),
      `.image` (typed), `.pts`, `.duration`, `.format`, `.size`.
- [ ] `GstCamera.v4l2(...)` / `GstCamera.rtsp(...)` factories (config builders).
- [ ] `VideoReader` (files: grab + seek/duration/fps/speed) and `VideoWriter`.
- [ ] `gstreamer` feature in `kornia-py/Cargo.toml` (`#[cfg(feature="gstreamer")]`,
      registered in the `capture` submodule).
- [ ] Zero-copy Image wrap of gst frames (already zero-copy in Rust via
      `GstResource`; wrap into `PyImageApi` with the keepalive).

## 4. Missing — GStreamer Rust `GstFrame` extensions

- [ ] `f32` formats (`GRAYF32`) → `to_image_f32::<C>()`.
- [ ] `GRAY16_BE` (byte-swap copy path) and more `u8` formats (`BGRA`, `RGBx`).
- [ ] `grab_raw()` — encoded/passthrough appsink (`image/jpeg`, `video/x-h264`)
      returning the mapped bytes + caps (GStreamer analogue of V4L `EncodedFrame`).
- [ ] Multi-plane formats (I420 / NV12) — expose planes / per-plane strides.
- [ ] Configurable ring depth (currently a fixed `FixedCircularBuffer<_, 5>`).

## 5. Missing — GPU / CUDA zero-copy (prototype proven → productize)

The full **GStreamer(NVMM) → CUDA → PyTorch** zero-copy chain is **proven on real
Jetson Orin hardware** with a native prototype (see §7 for the working recipe and
results: 1080p RGBA, 66.8 fps end-to-end, no host copy, `torch` CUDA tensor sharing
the GStreamer GPU frame). Productization into kornia:

- [ ] `Device`-domain `GstResource`: FFI to `NvBufSurface` +
      `NvBufSurfaceMapEglImage` + `cuGraphicsEGLRegisterImage` /
      `cuGraphicsResourceGetMappedEglFrame` → `CUdeviceptr`. Use the CUDA **primary
      context** so pointers are valid in cudarc/torch. (Jetson NVMM path.)
- [ ] Mainline `GstCudaMemory` path (dGPU / `nvcodec` `cudaupload`) as an
      alternative source of the `CUdeviceptr` (no EGL). Available on x86 + nvcodec.
- [ ] `GstFrame` `Device` variant carrying `MemoryDomain::Device { id }` /
      `Unified` + the CUDA pointer + keepalive.
- [ ] DLPack-CUDA export in `kornia-py` (`__dlpack__` with `device_type=kDLCUDA`,
      `device_id`, stream) → `torch.from_dlpack(...)` gives a zero-copy CUDA tensor.
      (`kornia-py` DLPack export already exists; extend to device tensors.)
- [ ] CUDA **stream synchronization** correctness (DLPack stream field) so torch
      reads completed frames.
- [ ] Keepalive via the DLPack deleter (unregister EGL / unmap NvBufSurface / unref
      sample when torch frees the tensor) — validated in the prototype.
- [ ] cudarc interop for a *borrowed* external `CUdeviceptr` (non-owning device
      resource) so kornia CUDA ops can run on the frame in-place.

## 6. Missing — release / CI / packaging

- [ ] **V4L**: verified self-contained (no `libv4l`, pure ioctls) → ships in the
      standard **manylinux** Linux wheel. CI must build Linux wheels **with
      `--features v4l`** (and macOS/Windows **without**). Linux-only; cross-arch OK
      (aarch64 untested — only x86_64 exercised here).
- [ ] **GStreamer**: no portable wheel (dlopens plugins + GLib). Ship as **opt-in
      sdist / source build** against the system GStreamer, or per-platform wheels
      (Jetson: JetPack GStreamer). ABI-stable across the 1.x series.
- [ ] Build without gstreamer-rs version features (`v1_18`, …) for max
      forward-compatibility.
- [ ] Decide abi3 vs per-version wheels (currently per-version; abi3 would collapse
      the matrix — separate effort, some risk with rust-numpy).

## 7. Proven GPU recipe (reference)

Validated on `nvidia-orin00` (Jetson Orin, JetPack R36.4.3, CUDA 12.6, torch 2.11):

```
videotestsrc → nvvidconv → video/x-raw(memory:NVMM),RGBA   (frame is GPU-resident)
  → gst_buffer_map → NvBufSurface
  → NvBufSurfaceMapEglImage → EGLImage
  → cuGraphicsEGLRegisterImage + cuGraphicsResourceGetMappedEglFrame → CUdeviceptr
      (pitch-linear; same physical memory, no copy)
  → DLManagedTensor (kDLCUDA, uint8, [H,W,4]; deleter releases EGL/surface/sample)
  → torch.utils.dlpack.from_dlpack → CUDA tensor sharing the GStreamer GPU frame
```

Result: `shape=(1080,1920,4) dtype=uint8 device=cuda:0`, on-device `mean` runs
directly on the frame, **60 frames @ 66.8 fps, no host download**. Cost eliminated
vs the download path: ~1.55 ms/frame H2D at 1080p (measured).

Key correctness notes (carry into the Rust port):
- Use `cuDevicePrimaryCtxRetain` (not `cuCtxCreate`) so the pointer is valid in
  torch's context.
- DLPack strides are in **elements**; the row stride comes from `CUeglFrame.pitch`.
- The DLPack **deleter is the keepalive** — releases EGL map + NvBufSurface + gst
  sample exactly when torch frees the tensor.
- Jetson NVMM is `NVBUF_MEM_SURFACE_ARRAY` → EGL mapping is required (no direct
  `dataPtr`). Mainline `GstCudaMemory` (dGPU) exposes a `CUdeviceptr` directly.
