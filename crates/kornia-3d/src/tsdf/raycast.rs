//! Raycasting: depth, vertex, normal and colour maps of a pinhole view,
//! in the layout of the RGB-D ICP's [`RgbdLevel`].

use kornia_image::{Image, ImageSize};
use kornia_oxide_kernels::tsdf_core::{self as tc, VoxelSampler};

use super::sensor::{check_host, check_image_size, pinhole_constants};
use super::volume::VolumeRef;
use super::workspace::TsdfWorkspace;
use super::{
    check_fp_env, check_pose, rigid_from_pose, BlockDim, StdOps, SupportedBlockDim, TsdfError,
    TsdfLayout, TsdfVolume,
};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use crate::registration::{RgbdLevel, RgbdPyramid};

/// Steps beyond `(depth_max - depth_min) / voxel_size` in the default cap.
const EXTRA_STEPS: f64 = 2.0;

/// Raycast parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastParams {
    /// March start, camera z in metres, `>= 0`.
    pub depth_min: f32,
    /// March end (exclusive), camera z in metres, `> depth_min`.
    pub depth_max: f32,
    /// Smallest weight of a voxel the ray may use, `>= 1`.
    pub weight_threshold: u16,
    /// Hard cap on march steps, `>= 1`.
    pub max_steps: u32,
    /// Also output colours.
    pub colors: bool,
}

impl RaycastParams {
    /// Weight threshold 1, no colour, and the step cap
    /// `ceil((depth_max - depth_min) / voxel_size) + 2`.
    ///
    /// # Arguments
    ///
    /// * `layout` - The volume's layout (voxel size).
    /// * `depth_min` - March start (metres), finite, `>= 0`.
    /// * `depth_max` - March end (metres), finite, `> depth_min`.
    ///
    /// # Returns
    ///
    /// The parameters.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for an invalid range.
    pub fn new(layout: &TsdfLayout, depth_min: f32, depth_max: f32) -> Result<Self, TsdfError> {
        let span = (depth_max as f64 - depth_min as f64) / layout.voxel_size() as f64;
        let steps = span.ceil() + EXTRA_STEPS;
        let p = Self {
            depth_min,
            depth_max,
            weight_threshold: 1,
            max_steps: if steps < u32::MAX as f64 {
                steps as u32
            } else {
                u32::MAX
            },
            colors: false,
        };
        p.validate()?;
        Ok(p)
    }

    pub(crate) fn validate(&self) -> Result<(), TsdfError> {
        let bad =
            |field: &'static str, value: f64| Err(TsdfError::InvalidParameter { field, value });
        if !(self.depth_min >= 0.0 && self.depth_min < f32::INFINITY) {
            return bad("depth_min", self.depth_min as f64);
        }
        if !(self.depth_max > self.depth_min && self.depth_max < f32::INFINITY) {
            return bad("depth_max", self.depth_max as f64);
        }
        if self.weight_threshold == 0 {
            return bad("weight_threshold", 0.0);
        }
        if self.max_steps == 0 {
            return bad("max_steps", 0.0);
        }
        Ok(())
    }
}

/// Raycast outputs of one view; pixels without a hit are zero.
#[derive(Clone)]
pub struct RaycastMaps {
    depth: Image<f32, 1>,
    vertices: Image<f32, 3>,
    normals: Image<f32, 3>,
    colors: Option<Image<u8, 3>>,
}

impl RaycastMaps {
    /// Zeroed maps.
    ///
    /// # Arguments
    ///
    /// * `size` - View size.
    /// * `colors` - Also hold a colour map.
    ///
    /// # Returns
    ///
    /// The maps.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for an oversized view,
    /// [`TsdfError::Image`] if an image cannot be allocated.
    pub fn new(size: ImageSize, colors: bool) -> Result<Self, TsdfError> {
        check_image_size(size)?;
        Ok(Self {
            depth: Image::from_size_val(size, 0.0)?,
            vertices: Image::from_size_val(size, 0.0)?,
            normals: Image::from_size_val(size, 0.0)?,
            colors: if colors {
                Some(Image::from_size_val(size, 0)?)
            } else {
                None
            },
        })
    }

    /// View size.
    pub fn size(&self) -> ImageSize {
        self.depth.size()
    }

    /// Camera z of the hit (metres), `0` = miss.
    pub fn depth(&self) -> &Image<f32, 1> {
        &self.depth
    }

    /// Camera-frame hit points (metres), `[0, 0, 0]` = miss.
    pub fn vertices(&self) -> &Image<f32, 3> {
        &self.vertices
    }

    /// Camera-frame unit normals facing the camera, `[0, 0, 0]` = invalid.
    pub fn normals(&self) -> &Image<f32, 3> {
        &self.normals
    }

    /// Colour of the hit voxel, when requested.
    pub fn colors(&self) -> Option<&Image<u8, 3>> {
        self.colors.as_ref()
    }
}

/// Output slices of one raycast.
pub(crate) struct RayOut<'a> {
    pub(crate) depth: Option<&'a mut [f32]>,
    pub(crate) vertices: &'a mut [f32],
    pub(crate) normals: &'a mut [f32],
    pub(crate) colors: Option<&'a mut [u8]>,
}

/// Core constants of a view of one volume.
pub(crate) fn core_params(
    layout: &TsdfLayout,
    log2: u32,
    k: &[f32; 6],
    volume_from_camera: &Pose3d,
    p: &RaycastParams,
) -> Result<tc::RaycastParams, TsdfError> {
    check_pose(volume_from_camera)?;
    let vfc = rigid_from_pose(volume_from_camera)?;
    let cfv = rigid_from_pose(&volume_from_camera.inverse())?;
    let vs = layout.voxel_size();
    Ok(tc::RaycastParams {
        volume_from_camera: vfc,
        camera_from_volume_r: cfv.r,
        cx: k[2],
        cy: k[3],
        inv_fx: k[4],
        inv_fy: k[5],
        depth_min: p.depth_min,
        depth_max: p.depth_max,
        voxel_size: vs,
        inv_voxel: layout.grid(log2).inv_voxel,
        trunc: layout.truncation(),
        block_step: ((1u32 << log2) as f32) * vs,
        weight_threshold: p.weight_threshold as u32,
        max_steps: p.max_steps,
    })
}

/// Raycasts `n` views (ascending submap order) into `out`: per pixel the
/// smallest hit `z` wins, the lower view on a tie.
pub(crate) fn raycast_views<'v>(
    size: ImageSize,
    rps: &[tc::RaycastParams],
    view: &dyn Fn(usize) -> VolumeRef<'v>,
    out: RayOut<'_>,
) {
    let RayOut {
        mut depth,
        vertices,
        normals,
        mut colors,
    } = out;
    let w = size.width;
    for i in 0..size.width * size.height {
        let (px, py) = ((i % w) as u32, (i / w) as u32);
        let mut best: Option<(tc::RayHit, usize)> = None;
        for (v, rp) in rps.iter().enumerate() {
            if let Some(h) = tc::raycast_pixel::<StdOps, _>(rp, &view(v), px, py) {
                if best.is_none_or(|(b, _)| h.z < b.z) {
                    best = Some((h, v));
                }
            }
        }
        let (z, vert, norm) = match &best {
            Some((h, _)) => (h.z, h.vertex, h.normal),
            None => (0.0, [0.0; 3], [0.0; 3]),
        };
        if let Some(d) = depth.as_deref_mut() {
            d[i] = z;
        }
        vertices[3 * i..3 * i + 3].copy_from_slice(&vert);
        normals[3 * i..3 * i + 3].copy_from_slice(&norm);
        if let Some(c) = colors.as_deref_mut() {
            let rgb = match &best {
                Some((h, v)) => hit_color(&view(*v), &rps[*v], h.point),
                None => [0; 3],
            };
            c[3 * i..3 * i + 3].copy_from_slice(&rgb);
        }
    }
}

/// Colour of the voxel containing `p` if it is valid, else zero.
fn hit_color(view: &VolumeRef<'_>, rp: &tc::RaycastParams, p: [f32; 3]) -> [u8; 3] {
    if !view.pools.color {
        return [0; 3];
    }
    let Some(g) = tc::voxel_from_point(p, rp.inv_voxel) else {
        return [0; 3];
    };
    match (view.sample(g), view.voxel_slot(g)) {
        (Some((_, w)), Some(i)) if w as u32 >= rp.weight_threshold => {
            tc::unpack_rgb(view.pools.rgbw[i])
        }
        _ => [0; 3],
    }
}

/// Shared checks of a raycast call; returns the camera constants.
pub(crate) fn check_raycast(
    camera: &PinholeCamera,
    size: ImageSize,
    params: &RaycastParams,
    has_color: bool,
) -> Result<[f32; 6], TsdfError> {
    check_fp_env()?;
    params.validate()?;
    if params.colors && !has_color {
        return Err(TsdfError::InvalidParameter {
            field: "colors",
            value: 1.0,
        });
    }
    check_image_size(size)?;
    pinhole_constants(camera)
}

/// Splits `maps` into output slices after checking their size and
/// residency.
pub(crate) fn maps_out<'a>(
    camera_size: ImageSize,
    params: &RaycastParams,
    maps: &'a mut RaycastMaps,
) -> Result<RayOut<'a>, TsdfError> {
    if maps.size() != camera_size {
        return Err(TsdfError::SizeMismatch {
            expected: camera_size,
            got: maps.size(),
        });
    }
    check_host(&maps.depth)?;
    check_host(&maps.vertices)?;
    check_host(&maps.normals)?;
    if let Some(c) = &maps.colors {
        check_host(c)?;
    }
    if params.colors && maps.colors.is_none() {
        return Err(TsdfError::InvalidParameter {
            field: "colors",
            value: 0.0,
        });
    }
    let colors = match maps.colors.as_mut() {
        Some(c) => {
            let s = c.as_slice_mut();
            if params.colors {
                Some(s)
            } else {
                s.fill(0);
                None
            }
        }
        None => None,
    };
    Ok(RayOut {
        depth: Some(maps.depth.as_slice_mut()),
        vertices: maps.vertices.as_slice_mut(),
        normals: maps.normals.as_slice_mut(),
        colors,
    })
}

/// Output slices of an ICP level (vertices and normals only).
pub(crate) fn level_out(level: &mut RgbdLevel) -> Result<RayOut<'_>, TsdfError> {
    check_host(&level.vertices)?;
    check_host(&level.normals)?;
    Ok(RayOut {
        depth: None,
        vertices: level.vertices.as_slice_mut(),
        normals: level.normals.as_slice_mut(),
        colors: None,
    })
}

impl<const B: usize> TsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    /// Raycasts a pinhole view of the volume.
    ///
    /// Each pixel's ray starts at camera z `depth_min` and marches: one
    /// block over an unallocated block, one voxel over voxels below the
    /// weight threshold, else `max(tsdf * truncation, voxel_size)`. The
    /// first `tsdf >= 0 -> tsdf < 0` crossing of consecutive valid samples
    /// is the hit, refined linearly. The normal is the trilinear gradient
    /// of the 8 surrounding voxels (zero unless all are valid), in the
    /// camera frame, facing the camera.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `camera` - Intrinsics of the view, without distortion; the maps
    ///   set its size.
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March range, weight threshold, step cap, colour.
    /// * `maps` - Output maps.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the maps hold the view.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidIntrinsics`], [`TsdfError::DistortedCamera`],
    /// [`TsdfError::InvalidTransform`], [`TsdfError::InvalidParameter`]
    /// (params, or colours requested without colour storage),
    /// [`TsdfError::FpEnvironment`], [`TsdfError::Image`].
    pub fn raycast_into(
        &self,
        ws: &mut TsdfWorkspace,
        camera: &PinholeCamera,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        maps: &mut RaycastMaps,
    ) -> Result<(), TsdfError> {
        let _ = ws;
        let size = maps.size();
        let k = check_raycast(camera, size, params, self.pools.color)?;
        let rp = core_params(&self.layout, Self::LOG2, &k, volume_from_camera, params)?;
        let out = maps_out(size, params, maps)?;
        let view = self.view();
        raycast_views(size, &[rp], &|_| view, out);
        Ok(())
    }

    /// Raycasts into an RGB-D ICP level (vertex and normal maps, at the
    /// level's camera), so frame-to-model ICP consumes the model view.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `level` - Level to refill, e.g. from [`RgbdLevel::zeroed`].
    ///
    /// # Returns
    ///
    /// `Ok(())` once the level holds the view.
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::raycast_into`].
    pub fn raycast_level_into(
        &self,
        ws: &mut TsdfWorkspace,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        level: &mut RgbdLevel,
    ) -> Result<(), TsdfError> {
        let _ = ws;
        let params = RaycastParams {
            colors: false,
            ..*params
        };
        let size = level.size();
        let k = check_raycast(level.camera(), size, &params, self.pools.color)?;
        let rp = core_params(&self.layout, Self::LOG2, &k, volume_from_camera, &params)?;
        let view = self.view();
        raycast_views(size, &[rp], &|_| view, level_out(level)?);
        Ok(())
    }

    /// Raycasts every level of an RGB-D ICP pyramid at its own camera.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `pyramid` - Pyramid to refill, e.g. from [`RgbdPyramid::zeroed`].
    ///
    /// # Returns
    ///
    /// `Ok(())` once every level holds the view.
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::raycast_into`].
    pub fn raycast_pyramid_into(
        &self,
        ws: &mut TsdfWorkspace,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        pyramid: &mut RgbdPyramid,
    ) -> Result<(), TsdfError> {
        for level in pyramid.levels.iter_mut() {
            self.raycast_level_into(ws, volume_from_camera, params, level)?;
        }
        Ok(())
    }
}
