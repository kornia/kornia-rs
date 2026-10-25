//! CUDA twins of [`unproject_depth_into`](super::unproject_depth_into) and
//! [`register_depth`](super::register_depth): device-resident images in,
//! device-resident points or depth out, no host sync per call.
//!
//! The kernels below are textual mirrors of `kernels.rs` compiled with
//! `fmad=false`, so the device output equals the host output exactly
//! (asserted by test).
//!
//! * Unprojection is three launches: per-row kept counts (one block per
//!   row), one exclusive scan over the rows (one block), and a per-row
//!   ordered compaction (warp ballots + a block scan per 256-pixel tile).
//!   The order of the points is the host's row-major order, not the
//!   arbitrary order an atomic append would give.
//! * Registration is a memset, one scatter (a thread per source pixel,
//!   `atomicMax` of the inverted depth key per splat cell) and a finalize
//!   pass. The scatter cannot write the final value directly because the
//!   z-buffer needs the inverted key to be `0` = empty. For `f32` the
//!   z-buffer is `dst` itself; `u16` uses a reusable `u32` scratch buffer
//!   (there is no 16-bit `atomicMax`).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, ValidAsZeroBits};
use kornia_image::{Image, ImageError};
use kornia_tensor::CudaKernel;

use super::kernels::{self, RegisterParams, UnprojectParams, KIND_F32};
use super::{check_same_size, DepthError, DepthSample};
use crate::camera::PinholeCamera;
use crate::pointcloud::PointCloud;
use crate::pose::Pose3d;

// SAFETY: plain-old-data, repr(C), only 4-byte fields (no padding); mirrored
// field-for-field by `struct Unp` in KERNELS.
unsafe impl DeviceRepr for UnprojectParams {}
// SAFETY: plain-old-data, repr(C), only 4-byte fields (no padding); mirrored
// field-for-field by `struct Reg` in KERNELS.
unsafe impl DeviceRepr for RegisterParams {}

/// Threads per row block of the unprojection kernels (`ROW_BLOCK` in KERNELS).
const ROW_BLOCK: u32 = 256;
/// Threads of the one-block row scan (`SCAN_BLOCK` in KERNELS).
const SCAN_BLOCK: u32 = 1024;
/// Scatter block: a warp per row, 8 rows.
const SCATTER_BLOCK: (u32, u32) = (32, 8);

const KERNELS: &str = r#"
#define ROW_BLOCK 256
#define ROW_WARPS (ROW_BLOCK / 32)
#define SCAN_BLOCK 1024
#define FULL 0xffffffffu

struct Unp { float inv_fx, inv_fy, cx, cy, scale; int width, height; };
struct Reg {
    Unp src;
    float inv_scale;
    float r[9];
    float t[3];
    float fx, fy, cx, cy;
    int width, height, radius;
};

/* ---- mirrors of kernels.rs; keep expression trees identical ---- */

/* DepthSample::metres */
__device__ __forceinline__ float metres(unsigned short d, float s) { return (float)d * s; }
__device__ __forceinline__ float metres(float d, float s) { return d * s; }
/* is_valid_metres */
__device__ __forceinline__ bool valid_z(float z) {
    return z > 0.0f && z < __int_as_float(0x7f800000);
}
/* mask non-zero and valid metres */
template <typename T>
__device__ __forceinline__ bool kept(const T* row, const unsigned char* mrow, int x, float s,
                                     float* z) {
    *z = metres(row[x], s);
    return valid_z(*z) && (mrow == 0 || mrow[x] != 0);
}

/* ---- unprojection ---- */

template <typename T>
__device__ void count_rows(const T* __restrict__ depth, const unsigned char* __restrict__ mask,
                           Unp p, unsigned int* __restrict__ row_count) {
    __shared__ unsigned int warp_sum[ROW_WARPS];
    int y = blockIdx.x;
    const T* row = depth + (size_t)y * p.width;
    const unsigned char* mrow = mask ? mask + (size_t)y * p.width : 0;
    unsigned int c = 0;
    for (int x = threadIdx.x; x < p.width; x += ROW_BLOCK) {
        float z;
        c += kept(row, mrow, x, p.scale, &z) ? 1u : 0u;
    }
    for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(FULL, c, off);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int s = 0;
        for (int i = 0; i < ROW_WARPS; ++i) s += warp_sum[i];
        row_count[y] = s;
    }
}

/* Exclusive scan of the row counts by one SCAN_BLOCK block: each thread sums
   a run of rows, warp 0 scans the SCAN_BLOCK partials (32 per lane + one
   shuffle scan), each thread then walks its run. Also writes the total. */
extern "C" __global__ void unproject_scan(const unsigned int* __restrict__ row_count, int rows,
                                          unsigned int* __restrict__ row_start,
                                          unsigned int* __restrict__ total) {
    __shared__ unsigned int part[SCAN_BLOCK];
    int t = threadIdx.x;
    int chunk = (rows + SCAN_BLOCK - 1) / SCAN_BLOCK;
    int b0 = t * chunk;
    int b1 = min(rows, b0 + chunk);
    unsigned int s = 0;
    for (int b = b0; b < b1; ++b) s += row_count[b];
    part[t] = s;
    __syncthreads();
    if (t < 32) {
        const int per = SCAN_BLOCK / 32;
        unsigned int acc = 0;
        for (int i = 0; i < per; ++i) {
            unsigned int v = part[t * per + i];
            part[t * per + i] = acc;
            acc += v;
        }
        unsigned int incl = acc;
        for (int off = 1; off < 32; off <<= 1) {
            unsigned int o = __shfl_up_sync(FULL, incl, off);
            if (t >= off) incl += o;
        }
        unsigned int excl = incl - acc;
        for (int i = 0; i < per; ++i) part[t * per + i] += excl;
        if (t == 31) *total = incl;
    }
    __syncthreads();
    unsigned int a = part[t];
    for (int b = b0; b < b1; ++b) {
        row_start[b] = a;
        a += row_count[b];
    }
}

/* Ordered compaction of one row: per 256-pixel tile, a ballot per warp gives
   each kept lane its rank in the warp, the warp totals give each warp its
   offset in the tile. Same order as kernels.rs::write_rows. */
template <typename T>
__device__ void write_rows(const T* __restrict__ depth, const unsigned char* __restrict__ mask,
                           Unp p, const unsigned int* __restrict__ row_start,
                           float* __restrict__ xyz, unsigned int* __restrict__ index) {
    __shared__ unsigned int warp_sum[ROW_WARPS];
    int y = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* row = depth + (size_t)y * p.width;
    const unsigned char* mrow = mask ? mask + (size_t)y * p.width : 0;
    float yn = ((float)y - p.cy) * p.inv_fy;
    unsigned int base = row_start[y];
    for (int x0 = 0; x0 < p.width; x0 += ROW_BLOCK) {
        int x = x0 + threadIdx.x;
        float z = 0.0f;
        bool ok = x < p.width && kept(row, mrow, x, p.scale, &z);
        unsigned int bal = __ballot_sync(FULL, ok);
        if (lane == 0) warp_sum[warp] = __popc(bal);
        __syncthreads();
        unsigned int before = 0, tile = 0;
        for (int i = 0; i < ROW_WARPS; ++i) {
            unsigned int c = warp_sum[i];
            before += i < warp ? c : 0u;
            tile += c;
        }
        if (ok) {
            size_t k = (size_t)base + before + __popc(bal & ((1u << lane) - 1u));
            float xn = ((float)x - p.cx) * p.inv_fx;
            xyz[3 * k] = xn * z;
            xyz[3 * k + 1] = yn * z;
            xyz[3 * k + 2] = z;
            if (index) index[k] = (unsigned int)y * (unsigned int)p.width + (unsigned int)x;
        }
        base += tile;
        __syncthreads();   /* warp_sum is rewritten by the next tile */
    }
}

extern "C" __global__ void unproject_count_u16(const unsigned short* depth,
                                               const unsigned char* mask, Unp p,
                                               unsigned int* row_count) {
    count_rows(depth, mask, p, row_count);
}
extern "C" __global__ void unproject_count_f32(const float* depth, const unsigned char* mask,
                                               Unp p, unsigned int* row_count) {
    count_rows(depth, mask, p, row_count);
}
extern "C" __global__ void unproject_write_u16(const unsigned short* depth,
                                               const unsigned char* mask, Unp p,
                                               const unsigned int* row_start, float* xyz,
                                               unsigned int* index) {
    write_rows(depth, mask, p, row_start, xyz, index);
}
extern "C" __global__ void unproject_write_f32(const float* depth, const unsigned char* mask,
                                               Unp p, const unsigned int* row_start,
                                               float* xyz, unsigned int* index) {
    write_rows(depth, mask, p, row_start, xyz, index);
}

/* ---- registration ---- */

/* kernels.rs::scatter_rows + splat_target + quantise_u16 / scale_f32 +
   splat_max. Keys are inverted values: nearer = larger, 0 = empty. */
template <typename T, bool U16>
__device__ void scatter(const T* __restrict__ src, Reg p, unsigned int* __restrict__ zb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.src.width || y >= p.src.height) return;
    float z = metres(src[(size_t)y * p.src.width + x], p.src.scale);
    if (!valid_z(z)) return;
    float px = (((float)x - p.src.cx) * p.src.inv_fx) * z;
    float py = (((float)y - p.src.cy) * p.src.inv_fy) * z;
    float qx = p.r[0] * px + p.r[1] * py + p.r[2] * z + p.t[0];
    float qy = p.r[3] * px + p.r[4] * py + p.r[5] * z + p.t[1];
    float qz = p.r[6] * px + p.r[7] * py + p.r[8] * z + p.t[2];
    if (isnan(qz) || qz <= 0.0f) return;
    float iz = 1.0f / qz;
    float u = p.fx * qx * iz + p.cx;
    float v = p.fy * qy * iz + p.cy;
    float uf = floorf(u + 0.5f);
    float vf = floorf(v + 0.5f);
    float rr = (float)p.radius;
    if (!(uf >= -rr && uf < (float)(p.width + p.radius) && vf >= -rr
          && vf < (float)(p.height + p.radius))) return;
    unsigned int key;
    if (U16) {
        float q = roundf(qz * p.inv_scale);
        if (!(q >= 1.0f && q <= 65534.0f)) return;
        key = ~(unsigned int)q;
    } else {
        float o = qz * p.inv_scale;
        if (!valid_z(o)) return;
        key = ~__float_as_uint(o);
    }
    int ui = (int)uf, vi = (int)vf;
    int x0 = max(ui - p.radius, 0), x1 = min(ui + p.radius, p.width - 1);
    int y0 = max(vi - p.radius, 0), y1 = min(vi + p.radius, p.height - 1);
    for (int yy = y0; yy <= y1; ++yy) {
        unsigned int* line = zb + (size_t)yy * p.width;
        for (int xx = x0; xx <= x1; ++xx) {
            /* Keys only grow: a stale read never skips a needed update. */
            if (key > line[xx]) atomicMax(line + xx, key);
        }
    }
}

extern "C" __global__ void register_scatter_u16(const unsigned short* src, Reg p,
                                                unsigned int* zb) {
    scatter<unsigned short, true>(src, p, zb);
}
extern "C" __global__ void register_scatter_f32(const float* src, Reg p, unsigned int* zb) {
    scatter<float, false>(src, p, zb);
}

/* In place over dst's bit patterns: key -> bits(value), empty stays 0. */
extern "C" __global__ void register_finalize_f32(unsigned int* __restrict__ buf,
                                                 unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int k = buf[i];
    buf[i] = k ? ~k : 0u;
}
extern "C" __global__ void register_finalize_u16(const unsigned int* __restrict__ zb,
                                                 unsigned short* __restrict__ dst,
                                                 unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int k = zb[i];
    dst[i] = k ? (unsigned short)(~k) : (unsigned short)0;
}
"#;

/// Kernel names, in the order [`CudaDepthProjector::new`] unpacks them.
const NAMES: [&str; 9] = [
    "unproject_count_u16",
    "unproject_count_f32",
    "unproject_write_u16",
    "unproject_write_f32",
    "unproject_scan",
    "register_scatter_u16",
    "register_scatter_f32",
    "register_finalize_u16",
    "register_finalize_f32",
];

fn cuda_err(e: impl std::fmt::Display) -> DepthError {
    DepthError::Image(ImageError::Cuda(e.to_string()))
}

/// The device slice of a device-resident image on device `ordinal`.
fn device_slice<T>(img: &Image<T, 1>, ordinal: usize) -> Result<&CudaSlice<T>, DepthError>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    check_device(img, ordinal)?;
    img.as_cudaslice()
        .ok_or_else(|| ImageError::HostResident.into())
}

/// `HostResident` for a host image (a routing bug, distinct from `Cuda`
/// driver failures), `DeviceMismatch` for an image on another device.
fn check_device<T>(img: &Image<T, 1>, ordinal: usize) -> Result<(), DepthError>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    let Some(stream) = img.cuda_stream() else {
        return Err(ImageError::HostResident.into());
    };
    if stream.context().ordinal() != ordinal {
        return Err(ImageError::DeviceMismatch.into());
    }
    Ok(())
}

/// Grows `buf` to at least `n` elements (contents unspecified).
fn ensure_len(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u32>,
    n: usize,
) -> Result<(), DepthError> {
    if buf.len() < n {
        *buf = stream.alloc_zeros::<u32>(n).map_err(cuda_err)?;
    }
    Ok(())
}

/// Device-resident unprojection output, caller-owned and reusable across
/// calls: `count` points of interleaved `x, y, z` (metres, `f32`) and,
/// optionally, the source pixel index of each.
pub struct CudaPointCloud {
    xyz: CudaSlice<f32>,
    pixel_index: Option<CudaSlice<u32>>,
    count: CudaSlice<u32>,
    capacity: usize,
    stream: Arc<CudaStream>,
}

impl CudaPointCloud {
    /// Point capacity.
    ///
    /// # Returns
    ///
    /// The number of points the buffers hold; an unprojection needs one
    /// slot per depth pixel.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Device points.
    ///
    /// # Returns
    ///
    /// The `3 × capacity` `f32` buffer; the first `3 × count` hold
    /// `x, y, z` per point, in the host's row-major order.
    pub fn xyz(&self) -> &CudaSlice<f32> {
        &self.xyz
    }

    /// Device source-pixel indices.
    ///
    /// # Returns
    ///
    /// The `capacity` `u32` buffer (`y · width + x` per point), or `None`
    /// when allocated without indices.
    pub fn pixel_index(&self) -> Option<&CudaSlice<u32>> {
        self.pixel_index.as_ref()
    }

    /// Device point count.
    ///
    /// # Returns
    ///
    /// One `u32` written by the last unprojection, for device consumers that
    /// must not sync.
    pub fn count(&self) -> &CudaSlice<u32> {
        &self.count
    }

    /// Reads the point count. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// The number of points of the last unprojection.
    ///
    /// # Errors
    ///
    /// [`DepthError::Image`]`(`[`ImageError::Cuda`]`)` on a copy failure.
    pub fn len_to_host(&self) -> Result<usize, DepthError> {
        let c = self.stream.clone_dtoh(&self.count).map_err(cuda_err)?;
        Ok((c.first().copied().unwrap_or(0) as usize).min(self.capacity))
    }

    /// Copies the points to a host [`PointCloud`]. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// The points widened to `f64`, identical to the host
    /// [`unproject_depth`](super::unproject_depth) of the same input.
    ///
    /// # Errors
    ///
    /// [`DepthError::Image`]`(`[`ImageError::Cuda`]`)` on a copy failure.
    pub fn to_host(&self) -> Result<PointCloud, DepthError> {
        let n = self.len_to_host()?;
        if n == 0 {
            return Ok(PointCloud::new(Vec::new(), None, None));
        }
        let xyz = self
            .stream
            .clone_dtoh(&self.xyz.slice(0..3 * n))
            .map_err(cuda_err)?;
        let points = xyz
            .chunks_exact(3)
            .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
            .collect();
        Ok(PointCloud::new(points, None, None))
    }

    /// Copies the source-pixel indices to the host. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// One index per point, or `None` when allocated without indices.
    ///
    /// # Errors
    ///
    /// [`DepthError::Image`]`(`[`ImageError::Cuda`]`)` on a copy failure.
    pub fn pixel_index_to_host(&self) -> Result<Option<Vec<u32>>, DepthError> {
        let Some(ix) = self.pixel_index.as_ref() else {
            return Ok(None);
        };
        let n = self.len_to_host()?;
        if n == 0 {
            return Ok(Some(Vec::new()));
        }
        Ok(Some(
            self.stream.clone_dtoh(&ix.slice(0..n)).map_err(cuda_err)?,
        ))
    }
}

/// CUDA depth unprojection and registration on one stream: the kernels are
/// compiled once at construction and the scratch buffers are reused across
/// calls. Work is enqueued without host syncs; read results after syncing
/// the stream.
///
/// Mid-run CUDA errors leave the stream state suspect; falling back to the
/// host functions is safe because they produce identical results.
pub struct CudaDepthProjector {
    stream: Arc<CudaStream>,
    k_count: [CudaKernel; 2],
    k_write: [CudaKernel; 2],
    k_scan: CudaKernel,
    k_scatter: [CudaKernel; 2],
    k_finalize: [CudaKernel; 2],
    row_count: CudaSlice<u32>,
    row_start: CudaSlice<u32>,
    zbuf: CudaSlice<u32>,
}

impl CudaDepthProjector {
    /// Compiles the kernels (nvrtc) for `stream`'s device and allocates the
    /// scratch buffers. Synchronizes `stream` once, so compile and
    /// allocation failures surface here rather than on the first frame.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream every call of this projector enqueues on.
    ///
    /// # Returns
    ///
    /// A projector ready for [`unproject_depth`](Self::unproject_depth) and
    /// [`register_depth`](Self::register_depth).
    ///
    /// # Errors
    ///
    /// [`DepthError::Image`]`(`[`ImageError::Cuda`]`)` on a compile or
    /// allocation failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudarc::driver::CudaContext;
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::CudaDepthProjector;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.default_stream();
    /// let mut projector = CudaDepthProjector::new(&stream)?;
    /// let camera = PinholeCamera {
    ///     fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
    ///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
    /// };
    /// let size = ImageSize { width: 640, height: 480 };
    /// let depth = Image::<u16, 1>::from_size_val(size, 1000)?.to_cuda(&stream)?;
    /// let mut cloud = projector.alloc_point_cloud(size.width * size.height, false)?;
    /// projector.unproject_depth(&depth, 0.001, &camera, None, &mut cloud)?;
    /// assert_eq!(cloud.to_host()?.len(), size.width * size.height);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, DepthError> {
        let kernels =
            CudaKernel::compile_many(stream.context(), KERNELS, &NAMES).map_err(cuda_err)?;
        let mut it = kernels.into_iter();
        let mut next = || {
            it.next()
                .ok_or_else(|| cuda_err("compile_many returned fewer kernels than requested"))
        };
        let k_count = [next()?, next()?];
        let k_write = [next()?, next()?];
        let k_scan = next()?;
        let k_scatter = [next()?, next()?];
        let k_finalize = [next()?, next()?];
        let row_count = stream.alloc_zeros::<u32>(1).map_err(cuda_err)?;
        let row_start = stream.alloc_zeros::<u32>(1).map_err(cuda_err)?;
        let zbuf = stream.alloc_zeros::<u32>(1).map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        Ok(Self {
            stream: stream.clone(),
            k_count,
            k_write,
            k_scan,
            k_scatter,
            k_finalize,
            row_count,
            row_start,
            zbuf,
        })
    }

    /// The stream this projector enqueues on.
    ///
    /// # Returns
    ///
    /// The stream given to [`new`](Self::new).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Allocates a reusable device point buffer on this projector's stream.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum points, at least the depth pixel count of the
    ///   images it will receive.
    /// * `with_pixel_index` - Also store the source pixel index per point.
    ///
    /// # Returns
    ///
    /// An empty [`CudaPointCloud`].
    ///
    /// # Errors
    ///
    /// [`DepthError::Image`]`(`[`ImageError::Cuda`]`)` on allocation failure.
    pub fn alloc_point_cloud(
        &self,
        capacity: usize,
        with_pixel_index: bool,
    ) -> Result<CudaPointCloud, DepthError> {
        let s = &self.stream;
        let n = capacity.max(1);
        let xyz_len = n
            .checked_mul(3)
            .ok_or_else(|| cuda_err("point capacity overflows usize"))?;
        let pixel_index = if with_pixel_index {
            Some(s.alloc_zeros::<u32>(n).map_err(cuda_err)?)
        } else {
            None
        };
        Ok(CudaPointCloud {
            xyz: s.alloc_zeros::<f32>(xyz_len).map_err(cuda_err)?,
            pixel_index,
            count: s.alloc_zeros::<u32>(1).map_err(cuda_err)?,
            capacity,
            stream: s.clone(),
        })
    }

    /// Device twin of [`unproject_depth_into`](super::unproject_depth_into):
    /// same points, same order, same pixel indices.
    ///
    /// # Arguments
    ///
    /// * `depth` - Device-resident `u16` or `f32` depth image.
    /// * `metres_per_unit` - Metres per depth unit.
    /// * `camera` - Pinhole intrinsics of the depth image (distortion ignored).
    /// * `mask` - Optional device-resident `u8` mask of the same size;
    ///   non-zero = keep.
    /// * `out` - Point buffer from [`alloc_point_cloud`](Self::alloc_point_cloud)
    ///   with at least one slot per depth pixel.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the work is enqueued; read `out` after syncing.
    ///
    /// # Errors
    ///
    /// * The argument errors of [`unproject_depth_into`](super::unproject_depth_into).
    /// * [`DepthError::CapacityTooSmall`] if `out` holds fewer points than
    ///   depth pixels; [`DepthError::StreamMismatch`] if `out` is from
    ///   another projector stream.
    /// * [`DepthError::ImageTooLarge`] above `u32::MAX` pixels.
    /// * [`DepthError::Image`] with [`ImageError::HostResident`] for a host
    ///   image, [`ImageError::DeviceMismatch`] for an image on another
    ///   device, [`ImageError::Cuda`] on a launch failure.
    pub fn unproject_depth<T>(
        &mut self,
        depth: &Image<T, 1>,
        metres_per_unit: f32,
        camera: &PinholeCamera,
        mask: Option<&Image<u8, 1>>,
        out: &mut CudaPointCloud,
    ) -> Result<(), DepthError>
    where
        T: DepthSample + DeviceRepr + ValidAsZeroBits + Default,
    {
        let (w, h) = (depth.width(), depth.height());
        let p = kernels::unproject_params(camera, metres_per_unit, w, h)?;
        if let Some(m) = mask {
            check_same_size(depth, m)?;
        }
        // Point counts and pixel indices are u32 on the device.
        kernels::check_pixel_index(w, h)?;
        let n = w * h;
        if n > u32::MAX as usize {
            return Err(DepthError::ImageTooLarge {
                width: w,
                height: h,
            });
        }
        if out.capacity < n {
            return Err(DepthError::CapacityTooSmall {
                capacity: out.capacity,
                needed: n,
            });
        }
        if !Arc::ptr_eq(&out.stream, &self.stream) {
            return Err(DepthError::StreamMismatch);
        }
        let ordinal = self.stream.context().ordinal();
        let d = device_slice(depth, ordinal)?;
        let m = match mask {
            Some(m) => Some(device_slice(m, ordinal)?),
            None => None,
        };
        let st = self.stream.clone();
        if n == 0 {
            st.memset_zeros(&mut out.count).map_err(cuda_err)?;
            return Ok(());
        }
        ensure_len(&st, &mut self.row_count, h)?;
        ensure_len(&st, &mut self.row_start, h)?;
        let k = T::KERNEL_INDEX;
        let null = 0u64;
        let rows = LaunchConfig {
            grid_dim: (h as u32, 1, 1),
            block_dim: (ROW_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let b = self.k_count[k].launch_builder(&st).arg(d);
        let b = match m {
            Some(m) => b.arg(m),
            None => b.arg(&null),
        };
        b.arg(&p)
            .arg(&mut self.row_count)
            .launch_cfg(rows)
            .map_err(cuda_err)?;

        self.k_scan
            .launch_builder(&st)
            .arg(&self.row_count)
            .arg(&p.height)
            .arg(&mut self.row_start)
            .arg(&mut out.count)
            .launch_cfg(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (SCAN_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(cuda_err)?;

        let b = self.k_write[k].launch_builder(&st).arg(d);
        let b = match m {
            Some(m) => b.arg(m),
            None => b.arg(&null),
        };
        let b = b.arg(&p).arg(&self.row_start).arg(&mut out.xyz);
        let b = match out.pixel_index.as_mut() {
            Some(ix) => b.arg(ix),
            None => b.arg(&null),
        };
        b.launch_cfg(rows).map_err(cuda_err)?;
        Ok(())
    }

    /// Device twin of [`register_depth`](super::register_depth): identical
    /// destination depth.
    ///
    /// # Arguments
    ///
    /// * `src` - Device-resident `u16` or `f32` source depth image.
    /// * `metres_per_unit` - Metres per depth unit of `src` and `dst`.
    /// * `src_camera` - Pinhole intrinsics of `src` (distortion ignored).
    /// * `dst_from_src` - Rigid transform from the source to the
    ///   destination camera frame.
    /// * `dst_camera` - Pinhole intrinsics of the destination grid.
    /// * `splat_radius` - Splat radius, `0` to [`super::MAX_SPLAT_RADIUS`].
    /// * `dst` - Device-resident destination image; fully overwritten.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the work is enqueued; read `dst` after syncing.
    ///
    /// # Errors
    ///
    /// * The argument errors of [`register_depth`](super::register_depth).
    /// * [`DepthError::ImageTooLarge`] above `u32::MAX` destination pixels.
    /// * [`DepthError::Image`] with [`ImageError::HostResident`] for a host
    ///   image, [`ImageError::DeviceMismatch`] for an image on another
    ///   device, [`ImageError::Cuda`] on a launch failure.
    #[allow(clippy::too_many_arguments)]
    pub fn register_depth<T>(
        &mut self,
        src: &Image<T, 1>,
        metres_per_unit: f32,
        src_camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst: &mut Image<T, 1>,
    ) -> Result<(), DepthError>
    where
        T: DepthSample + DeviceRepr + ValidAsZeroBits + Default,
    {
        let p = kernels::register_params(
            src_camera,
            metres_per_unit,
            dst_from_src,
            dst_camera,
            splat_radius,
            src.size(),
            dst.size(),
        )?;
        let n = dst.width() * dst.height();
        if n > u32::MAX as usize {
            return Err(DepthError::ImageTooLarge {
                width: dst.width(),
                height: dst.height(),
            });
        }
        let ordinal = self.stream.context().ordinal();
        let s = device_slice(src, ordinal)?;
        check_device(dst, ordinal)?;
        if n == 0 {
            return Ok(());
        }
        let d = dst
            .as_cudaslice_mut()
            .ok_or(DepthError::Image(ImageError::HostResident))?;
        let st = self.stream.clone();
        let k = T::KERNEL_INDEX;
        let n_u64 = n as u64;
        let (bw, bh) = SCATTER_BLOCK;
        let (sw, sh) = (p.src.width as u32, p.src.height as u32);
        let scatter_cfg = LaunchConfig {
            grid_dim: (sw.div_ceil(bw), sh.div_ceil(bh), 1),
            block_dim: (bw, bh, 1),
            shared_mem_bytes: 0,
        };
        let scatter_on = sw > 0 && sh > 0;

        if k == KIND_F32 {
            // `dst`'s own bits are the z-buffer.
            st.memset_zeros(d).map_err(cuda_err)?;
            if scatter_on {
                self.k_scatter[k]
                    .launch_builder(&st)
                    .arg(s)
                    .arg(&p)
                    .arg(&mut *d)
                    .launch_cfg(scatter_cfg)
                    .map_err(cuda_err)?;
            }
            self.k_finalize[k]
                .launch_builder(&st)
                .arg(&mut *d)
                .arg(&n_u64)
                .launch_1d(n as u32)
                .map_err(cuda_err)?;
        } else {
            ensure_len(&st, &mut self.zbuf, n)?;
            st.memset_zeros(&mut self.zbuf.slice_mut(0..n))
                .map_err(cuda_err)?;
            if scatter_on {
                self.k_scatter[k]
                    .launch_builder(&st)
                    .arg(s)
                    .arg(&p)
                    .arg(&mut self.zbuf)
                    .launch_cfg(scatter_cfg)
                    .map_err(cuda_err)?;
            }
            self.k_finalize[k]
                .launch_builder(&st)
                .arg(&self.zbuf)
                .arg(&mut *d)
                .arg(&n_u64)
                .launch_1d(n as u32)
                .map_err(cuda_err)?;
        }
        Ok(())
    }
}
