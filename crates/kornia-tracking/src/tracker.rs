use glam::{Mat2, Mat3, Vec2, Vec3};
use rayon::prelude::*;
use std::collections::HashMap;

// Define helper types and functions for affine operations using Mat3.
fn set_translation(mat: &mut Mat3, x: f32, y: f32) {
	// Preserve rotation (first two columns) and set translation (third column)
	*mat = Mat3::from_cols(mat.x_axis, mat.y_axis, Vec3::new(x, y, 1.0));
}
fn get_translation(mat: &Mat3) -> Vec2 {
	// translation is stored in the third column (x, y)
	Vec2::new(mat.z_axis.x, mat.z_axis.y)
}

// For patch residual computations, define simple aliases and helper functions.
type Mat2x52 = [[f32; 52]; 2];
fn mul_mat2_mat2x52(mat: Mat2, patt: Mat2x52) -> Mat2x52 {
	let mut res = [[0.0; 52]; 2];
	for j in 0..52 {
		let col = Vec2::new(patt[0][j], patt[1][j]);
		let prod = mat * col;
		res[0][j] = prod.x;
		res[1][j] = prod.y;
	}
	res
}
fn add_translation_to_mat2x52(mut patt: Mat2x52, trans: Vec2) -> Mat2x52 {
	for j in 0..52 {
		patt[0][j] += trans.x;
		patt[1][j] += trans.y;
	}
	patt
}
// Helper norm for a fixed-size vector stored as a slice.
fn vec_norm(vec: &[f32]) -> f32 {
	vec.iter().map(|x| x*x).sum::<f32>().sqrt()
}

#[derive(Default)]
pub struct PatchTracker<const N: u32> {
    last_keypoint_id: usize,
    tracked_points_map: HashMap<usize, Mat3>,
    previous_image_pyramid: Vec<image::GrayImage>,
}
impl<const LEVELS: u32> PatchTracker<LEVELS> {
    pub fn process_frame(&mut self, greyscale_image: &image::GrayImage) {
        let current_image_pyramid: Vec<image::GrayImage> = build_image_pyramid(greyscale_image, LEVELS);

        if !self.previous_image_pyramid.is_empty() {
            log::info!("old points {}", self.tracked_points_map.len());
            self.tracked_points_map = track_points::<LEVELS>(
                &self.previous_image_pyramid,
                &current_image_pyramid,
                &self.tracked_points_map,
            );
            log::info!("tracked old points {}", self.tracked_points_map.len());
        }
        // add new points
        let new_points = add_points(&self.tracked_points_map, greyscale_image);
        for point in &new_points {
            let mut v = Mat3::IDENTITY;
            set_translation(&mut v, point.x as f32, point.y as f32);
            self.tracked_points_map.insert(self.last_keypoint_id, v);
            self.last_keypoint_id += 1;
        }
        self.previous_image_pyramid = current_image_pyramid;
    }
    pub fn get_track_points(&self) -> HashMap<usize, (f32, f32)> {
        self.tracked_points_map
            .iter()
            .map(|(k, v)| (*k, { let t = get_translation(v); (t.x, t.y) }))
            .collect()
    }
    pub fn remove_id(&mut self, ids: &[usize]) {
        for id in ids {
            self.tracked_points_map.remove(id);
        }
    }
}

#[derive(Default)]
pub struct StereoPatchTracker<const N: u32> {
    last_keypoint_id: usize,
    tracked_points_map_cam0: HashMap<usize, Mat3>,
    previous_image_pyramid0: Vec<image::GrayImage>,
    tracked_points_map_cam1: HashMap<usize, Mat3>,
    previous_image_pyramid1: Vec<image::GrayImage>,
}

impl<const LEVELS: u32> StereoPatchTracker<LEVELS> {
    pub fn process_frame(&mut self, greyscale_image0: &image::GrayImage, greyscale_image1: &image::GrayImage) {
        let current_image_pyramid0: Vec<image::GrayImage> = build_image_pyramid(greyscale_image0, LEVELS);
        let current_image_pyramid1: Vec<image::GrayImage> = build_image_pyramid(greyscale_image1, LEVELS);
        if !self.previous_image_pyramid0.is_empty() {
            log::info!("old points {}", self.tracked_points_map_cam0.len());
            self.tracked_points_map_cam0 = track_points::<LEVELS>(
                &self.previous_image_pyramid0,
                &current_image_pyramid0,
                &self.tracked_points_map_cam0,
            );
            self.tracked_points_map_cam1 = track_points::<LEVELS>(
                &self.previous_image_pyramid1,
                &current_image_pyramid1,
                &self.tracked_points_map_cam1,
            );
            log::info!("tracked old points {}", self.tracked_points_map_cam0.len());
        }
        let new_points0 = add_points(&self.tracked_points_map_cam0, greyscale_image0);
        let tmp_tracked_points0: HashMap<usize, _> = new_points0
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let mut v = Mat3::IDENTITY;
                set_translation(&mut v, point.x as f32, point.y as f32);
                (i, v)
            })
            .collect();
        let tmp_tracked_points1 = track_points::<LEVELS>(
            &current_image_pyramid0,
            &current_image_pyramid1,
            &tmp_tracked_points0,
        );
        for (key0, pt0) in tmp_tracked_points0 {
            if let Some(pt1) = tmp_tracked_points1.get(&key0) {
                self.tracked_points_map_cam0.insert(self.last_keypoint_id, pt0);
                self.tracked_points_map_cam1.insert(self.last_keypoint_id, *pt1);
                self.last_keypoint_id += 1;
            }
        }
        self.previous_image_pyramid0 = current_image_pyramid0;
        self.previous_image_pyramid1 = current_image_pyramid1;
    }
    pub fn get_track_points(&self) -> [HashMap<usize, (f32, f32)>; 2] {
        let tracked_pts0 = self
            .tracked_points_map_cam0
            .iter()
            .map(|(k, v)| (*k, { let t = get_translation(v); (t.x, t.y) }))
            .collect();
        let tracked_pts1 = self
            .tracked_points_map_cam1
            .iter()
            .map(|(k, v)| (*k, { let t = get_translation(v); (t.x, t.y) }))
            .collect();
        [tracked_pts0, tracked_pts1]
    }
    pub fn remove_id(&mut self, ids: &[usize]) {
        for id in ids {
            self.tracked_points_map_cam0.remove(id);
            self.tracked_points_map_cam1.remove(id);
        }
    }
}

fn build_image_pyramid(greyscale_image: &image::GrayImage, levels: u32) -> Vec<image::GrayImage> {
    const FILTER_TYPE: image::imageops::FilterType = image::imageops::FilterType::Triangle;
    let (w0, h0) = greyscale_image.dimensions();
    (0..levels)
        .into_par_iter()
        .map(|i| {
            let scale_down: u32 = 1 << i;
            let (new_w, new_h) = (w0 / scale_down, h0 / scale_down);
            image::imageops::resize(greyscale_image, new_w, new_h, FILTER_TYPE)
        })
        .collect()
}

fn add_points(
    tracked_points_map: &HashMap<usize, Mat3>,
    grayscale_image: &image::GrayImage,
) -> Vec<patch::Corner> {
    const GRID_SIZE: u32 = 50;
    let num_points_in_cell = 1;
    let current_corners: Vec<patch::Corner> = tracked_points_map
        .values()
        .map(|v| {
            let t = get_translation(v);
            patch::Corner::new(t.x.round() as u32, t.y.round() as u32, 0.0)
        })
        .collect();
    image_utilities::detect_key_points(
        grayscale_image,
        GRID_SIZE,
        &current_corners,
        num_points_in_cell,
    )
}

fn track_points<const LEVELS: u32>(
    image_pyramid0: &[image::GrayImage],
    image_pyramid1: &[image::GrayImage],
    transform_maps0: &HashMap<usize, Mat3>,
) -> HashMap<usize, Mat3> {
    transform_maps0
        .par_iter()
        .filter_map(|(k, v)| {
            if let Some(new_v) = track_one_point::<LEVELS>(image_pyramid0, image_pyramid1, v) {
                if let Some(old_v) = track_one_point::<LEVELS>(image_pyramid1, image_pyramid0, &new_v) {
                    let diff = get_translation(v) - get_translation(&old_v);
                    if diff.length_squared() < 0.4 {
                        return Some((*k, new_v));
                    }
                }
            }
            None
        })
        .collect()
}

fn track_one_point<const LEVELS: u32>(
    image_pyramid0: &[image::GrayImage],
    image_pyramid1: &[image::GrayImage],
    transform0: &Mat3,
) -> Option<Mat3> {
    let mut patch_valid = true;
    let mut transform1 = *transform0;
    for i in (0..LEVELS).rev() {
        let scale_down = 1 << i;
        // Scale translation down
        let t = get_translation(&transform1) / scale_down as f32;
        set_translation(&mut transform1, t.x, t.y);

        let pattern = patch::Pattern52::new(
            &image_pyramid0[i as usize],
            get_translation(transform0).x / scale_down as f32,
            get_translation(transform0).y / scale_down as f32,
        );
        patch_valid &= pattern.valid;
        if patch_valid {
            patch_valid &= track_point_at_level(&image_pyramid1[i as usize], &pattern, &mut transform1);
            if !patch_valid {
                return None;
            }
        } else {
            return None;
        }
        // Scale translation back up
        let t_new = get_translation(&transform1) * scale_down as f32;
        set_translation(&mut transform1, t_new.x, t_new.y);
    }
    let new_r_mat = *transform0 * transform1;
    // Update rotation from the new matrix (preserving translation from transform1)
    transform1 = Mat3::from_cols(new_r_mat.x_axis, new_r_mat.y_axis, transform1.z_axis);
    Some(transform1)
}

pub fn track_point_at_level(
    grayscale_image: &image::GrayImage,
    dp: &patch::Pattern52,
    transform: &mut Mat3,
) -> bool {
    let optical_flow_max_iterations = 5;
    // Create a 2x52 pattern matrix (using dp.PATTERN_RAW divided by scale)
    // Note: Adjust the following as needed based on the actual type of dp.PATTERN_RAW
    let mut patt: Mat2x52 = {
        // ...existing code to initialize patt...
        dp.PATTERN_RAW.map(|row| row.map(|elem| elem / dp.pattern_scale_down))
    };
    for _iteration in 0..optical_flow_max_iterations {
        // Extract the rotation part from transform as a 2x2 matrix
        let rot = Mat2::from_cols(transform.x_axis.truncate(), transform.y_axis.truncate());
        let transformed_pat = add_translation_to_mat2x52(mul_mat2_mat2x52(rot, patt), get_translation(transform));
        if let Some(res) = dp.residual(grayscale_image, &transformed_pat) {
            let inc = -dp.h_se2_inv_j_se2_t * res;
            if !inc.iter().all(|x| x.is_finite()) {
                return false;
            }
            if inc.length() > 1e6 {
                return false;
            }
            let new_trans = *transform * image_utilities::se2_exp_matrix(&inc);
            *transform = new_trans;
            if !image_utilities::inbound(
                grayscale_image,
                get_translation(transform).x,
                get_translation(transform).y,
                2,
            ) {
                return false;
            }
        }
    }
    true
}
