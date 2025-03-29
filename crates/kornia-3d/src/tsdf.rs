// Truncated Signed Distance Function (TSDF) Volume Integration
//
// This module implements a volumetric 3D reconstruction technique
// using the Truncated Signed Distance Function (TSDF) approach,
// which represents surfaces implicitly as the zero-level set of a
// signed distance field truncated to a certain band around the surface.

use std::collections::HashMap;
use rayon::prelude::*;
use rayon::iter::IntoParallelIterator;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use std::io;
use std::sync::Mutex;

// Import the Marching Cubes lookup tables
use crate::tsdf_tables::{MC_EDGE_TABLE, MC_TRIANGLE_TABLE};

/// Errors that can occur during TSDF operations
#[derive(Error, Debug)]
pub enum TSDFError {
    /// Invalid input dimensions or parameters
    #[error("Invalid dimensions or parameters: {0}")]
    InvalidInput(String),

    /// Invalid camera pose or intrinsics
    #[error("Invalid camera parameters: {0}")]
    InvalidCamera(String),

    /// Error during mesh extraction
    #[error("Mesh extraction error: {0}")]
    MeshExtractionError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
}

/// TSDF Volume structure representing a 3D grid of voxels with truncated signed distances
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TSDFVolume {
    /// Physical size of each voxel in meters
    voxel_size: f32,
    
    /// Maximum distance for TSDF truncation in meters
    truncation_distance: f32,
    
    /// Voxel grid dimensions [x, y, z]
    dims: [usize; 3],
    
    /// World coordinates of the volume's minimum corner in meters
    origin: [f32; 3],

    /// TSDF values (-1 to 1) for each voxel
    tsdf: Vec<f32>,
    
    /// Integration weights for each voxel
    weights: Vec<f32>,
}

impl TSDFVolume {
    /// Create a new TSDF volume with the given parameters
    ///
    /// # Arguments
    /// * `voxel_size` - Physical size of each voxel in meters
    /// * `truncation_distance` - Maximum distance for TSDF truncation in meters
    /// * `dims` - Voxel grid dimensions [x, y, z]
    /// * `origin` - World coordinates of the volume's minimum corner in meters
    ///
    /// # Returns
    /// A new TSDFVolume instance initialized with the given parameters
    pub fn new(
        voxel_size: f32,
        truncation_distance: f32,
        dims: [usize; 3],
        origin: [f32; 3],
    ) -> Result<Self, TSDFError> {
        // Validate inputs
        if voxel_size <= 0.0 {
            return Err(TSDFError::InvalidInput("Voxel size must be positive".to_string()));
        }
        
        if truncation_distance <= 0.0 {
            return Err(TSDFError::InvalidInput("Truncation distance must be positive".to_string()));
        }
        
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err(TSDFError::InvalidInput("Dimensions must be non-zero".to_string()));
        }
        
        let total_voxels = dims[0] * dims[1] * dims[2];
        
        Ok(TSDFVolume {
            voxel_size,
            truncation_distance: truncation_distance.abs(), // Ensure positive
            dims,
            origin,
            tsdf: vec![1.0; total_voxels], // 1.0 = empty space
            weights: vec![0.0; total_voxels],
        })
    }

    /// Integrate a depth frame into the TSDF volume
    ///
    /// # Arguments
    /// * `depth_image` - Depth image data (row-major, in meters)
    /// * `width` - Width of the depth image in pixels
    /// * `height` - Height of the depth image in pixels
    /// * `intrinsics` - Camera intrinsic parameters
    /// * `camera_pose` - 4x4 transformation matrix from camera to world coordinates
    ///
    /// # Returns
    /// `Ok(())` if integration was successful, or an error otherwise
    pub fn integrate(
        &mut self,
        depth_image: &[f32],
        width: usize,
        height: usize,
        intrinsics: CameraIntrinsics,
        camera_pose: [[f32; 4]; 4],
    ) -> Result<(), TSDFError> {
        // Validate inputs
        if depth_image.len() != width * height {
            return Err(TSDFError::InvalidInput(
                format!("Depth image size ({}) does not match dimensions ({}x{}={})",
                        depth_image.len(), width, height, width * height)
            ));
        }

        // Precompute inverse camera pose (world-to-camera)
        let inv_pose = invert_transform_matrix(camera_pose);

        // Create a thread-safe vector for updates
        let updates = Mutex::new(Vec::new());
        
        // Process all voxels and collect updates
        (0..self.dims[2]).into_par_iter().for_each(|z| {
            let mut local_updates = Vec::new();
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    // 1. Compute voxel's world position (center)
                    let world_pos = [
                        self.origin[0] + (x as f32 + 0.5) * self.voxel_size,
                        self.origin[1] + (y as f32 + 0.5) * self.voxel_size,
                        self.origin[2] + (z as f32 + 0.5) * self.voxel_size,
                    ];

                    // 2. Transform to camera coordinates
                    let camera_pos = transform_point(inv_pose, world_pos);
                    
                    // Skip if point is behind camera
                    if camera_pos[2] <= 0.0 {
                        continue;
                    }

                    // 3. Project to pixel coordinates
                    let u = (intrinsics.fx * camera_pos[0] / camera_pos[2] + intrinsics.cx).round() as i32;
                    let v = (intrinsics.fy * camera_pos[1] / camera_pos[2] + intrinsics.cy).round() as i32;

                    // 4. Check if projected point is within image bounds
                    if u < 0 || v < 0 || u >= width as i32 || v >= height as i32 {
                        continue;
                    }

                    // 5. Get measured depth value from image
                    let depth = depth_image[v as usize * width + u as usize];
                    
                    // Skip invalid depth values
                    if depth <= 0.0 || !depth.is_finite() {
                        continue;
                    }

                    // 6. Compute signed distance
                    let sdf = depth - camera_pos[2]; // Positive = outside surface
                    
                    // Only update voxels within the truncation band
                    if sdf >= -self.truncation_distance {
                        // Truncate and normalize SDF
                        let tsdf_val = (sdf / self.truncation_distance).clamp(-1.0, 1.0);
                        
                        // Use a distance-based weighting scheme
                        // Weight decreases as distance from surface increases
                        let weight = if sdf.abs() < self.truncation_distance {
                            1.0 - sdf.abs() / self.truncation_distance
                        } else {
                            0.0
                        };

                        // Skip update if weight is too small
                        if weight < 1e-5 {
                            continue;
                        }

                        // Get voxel index and store update
                        let idx = self.voxel_index(x, y, z);
                        local_updates.push((idx, tsdf_val, weight));
                    }
                }
            }
            
            // ThreadLocal updates are pushed to global updates
            if !local_updates.is_empty() {
                let mut global_updates = updates.lock().unwrap();
                global_updates.extend(local_updates);
            }
        });
        
        // Apply all updates at once
        let updates = updates.into_inner().unwrap();
        for (idx, tsdf_val, weight) in updates {
            let old_weight = self.weights[idx];
            let new_weight = old_weight + weight;
            
            if new_weight > 0.0 {
                // Weighted average of TSDF values
                self.tsdf[idx] = (self.tsdf[idx] * old_weight + tsdf_val * weight) / new_weight;
                self.weights[idx] = new_weight;
            }
        }

        Ok(())
    }

    /// Get the total number of voxels in the volume
    pub fn volume_size(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    
    /// Get the physical dimensions of the volume in meters
    pub fn physical_size(&self) -> [f32; 3] {
        [
            self.dims[0] as f32 * self.voxel_size,
            self.dims[1] as f32 * self.voxel_size,
            self.dims[2] as f32 * self.voxel_size,
        ]
    }
    
    /// Get the dimensions of the volume in voxels
    pub fn dimensions(&self) -> [usize; 3] {
        self.dims
    }
    
    /// Get the voxel size in meters
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }
    
    /// Get the truncation distance in meters
    pub fn truncation_distance(&self) -> f32 {
        self.truncation_distance
    }
    
    /// Get the origin of the volume in world coordinates
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }
    
    /// Reset the TSDF volume to its initial state
    pub fn reset(&mut self) {
        self.tsdf.fill(1.0);
        self.weights.fill(0.0);
    }
    
    /// Helper: Compute flat index for 3D voxel grid
    fn voxel_index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }
    
    /// Get TSDF value at a specific voxel coordinate
    pub fn get_tsdf(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 1.0; // Out of bounds, return empty space
        }
        let idx = self.voxel_index(x, y, z);
        self.tsdf[idx]
    }
    
    /// Get weight value at a specific voxel coordinate
    pub fn get_weight(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 0.0; // Out of bounds
        }
        let idx = self.voxel_index(x, y, z);
        self.weights[idx]
    }
    
    /// Get a slice of the raw TSDF data
    pub fn tsdf_data(&self) -> &[f32] {
        &self.tsdf
    }
    
    /// Get a slice of the raw weight data
    pub fn weight_data(&self) -> &[f32] {
        &self.weights
    }
    
    /// Get mutable access to the TSDF data
    pub fn tsdf_data_mut(&mut self) -> &mut [f32] {
        &mut self.tsdf
    }
    
    /// Get mutable access to the weight data
    pub fn weight_data_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Extract a triangle mesh from the TSDF volume using Marching Cubes
    ///
    /// # Returns
    /// A tuple containing the vertices (positions) and triangles (indices)
    pub fn extract_mesh(&self) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>), TSDFError> {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let mut edge_cache = HashMap::new();
        
        // Process each cube in the volume (a cube consists of 8 neighboring voxels)
        for z in 0..self.dims[2] - 1 {
            for y in 0..self.dims[1] - 1 {
                for x in 0..self.dims[0] - 1 {
                    // Get TSDF values at the 8 corners of the cube
                    let corners = [
                        self.get_tsdf(x, y, z),
                        self.get_tsdf(x + 1, y, z),
                        self.get_tsdf(x + 1, y + 1, z),
                        self.get_tsdf(x, y + 1, z),
                        self.get_tsdf(x, y, z + 1),
                        self.get_tsdf(x + 1, y, z + 1),
                        self.get_tsdf(x + 1, y + 1, z + 1),
                        self.get_tsdf(x, y + 1, z + 1),
                    ];
                    
                    // Compute case index (which corners are inside/outside)
                    let mut case_idx = 0;
                    for (i, &val) in corners.iter().enumerate() {
                        if val <= 0.0 {
                            case_idx |= 1 << i;
                        }
                    }
                    
                    // Skip empty or full cubes
                    if case_idx == 0 || case_idx == 255 {
                        continue;
                    }
                    
                    // Process the triangles for this cube based on the case index
                    let _edge_indices = MC_EDGE_TABLE[case_idx];
                    
                    // Marching cubes triangle table gives us the edges where triangles should be created
                    let tri_table = &MC_TRIANGLE_TABLE[case_idx];
                    
                    let mut i = 0;
                    while tri_table[i] != -1 {
                        let mut triangle = [0; 3];
                        
                        // Process each vertex of the triangle
                        for j in 0..3 {
                            // Get the edge index
                            let edge = tri_table[i + j] as usize;
                            
                            // The edge connects two corners of the cube
                            let (v1, v2) = MC_EDGES[edge];
                            
                            // Create a unique key for this edge within the overall grid
                            let cube_key = (x, y, z);
                            let edge_key = (cube_key, edge);
                            
                            // Check if we've already processed this edge
                            if let Some(&vertex_idx) = edge_cache.get(&edge_key) {
                                triangle[j] = vertex_idx;
                            } else {
                                // Interpolate the zero-crossing point along this edge
                                let vertex = interpolate_vertex(
                                    corners[v1], 
                                    corners[v2],
                                    [x, y, z],
                                    v1,
                                    v2,
                                    self.voxel_size,
                                    self.origin,
                                );
                                
                                // Add the new vertex
                                vertices.push(vertex);
                                let vertex_idx = vertices.len() as u32 - 1;
                                
                                // Cache it for reuse
                                edge_cache.insert(edge_key, vertex_idx);
                                triangle[j] = vertex_idx;
                            }
                        }
                        
                        // Add the triangle
                        triangles.push([triangle[0], triangle[1], triangle[2]]);
                        i += 3;
                    }
                }
            }
        }
        
        Ok((vertices, triangles))
    }
}

/// Interpolate a vertex position along an edge where the TSDF crosses zero
fn interpolate_vertex(
    tsdf1: f32,
    tsdf2: f32,
    base_pos: [usize; 3],
    corner1: usize,
    corner2: usize,
    voxel_size: f32,
    origin: [f32; 3],
) -> [f32; 3] {
    // Avoid division by zero
    if (tsdf1 - tsdf2).abs() < 1e-6 {
        // Just use the midpoint if the values are very close
        let p1 = corner_position(corner1, base_pos);
        let p2 = corner_position(corner2, base_pos);
        return [
            origin[0] + (p1[0] + p2[0]) * 0.5 * voxel_size,
            origin[1] + (p1[1] + p2[1]) * 0.5 * voxel_size,
            origin[2] + (p1[2] + p2[2]) * 0.5 * voxel_size,
        ];
    }
    
    // Linear interpolation weight (find point where TSDF = 0)
    let mut t = tsdf1 / (tsdf1 - tsdf2);
    
    // Ensure t is between 0 and 1
    t = t.clamp(0.0, 1.0);
    
    // Get the positions of the corners
    let p1 = corner_position(corner1, base_pos);
    let p2 = corner_position(corner2, base_pos);
    
    // Interpolate position
    [
        origin[0] + (p1[0] + t * (p2[0] - p1[0])) * voxel_size,
        origin[1] + (p1[1] + t * (p2[1] - p1[1])) * voxel_size,
        origin[2] + (p1[2] + t * (p2[2] - p1[2])) * voxel_size,
    ]
}

/// Get the position of a cube corner based on its index
fn corner_position(corner: usize, base_pos: [usize; 3]) -> [f32; 3] {
    match corner {
        0 => [base_pos[0] as f32, base_pos[1] as f32, base_pos[2] as f32],
        1 => [base_pos[0] as f32 + 1.0, base_pos[1] as f32, base_pos[2] as f32],
        2 => [base_pos[0] as f32 + 1.0, base_pos[1] as f32 + 1.0, base_pos[2] as f32],
        3 => [base_pos[0] as f32, base_pos[1] as f32 + 1.0, base_pos[2] as f32],
        4 => [base_pos[0] as f32, base_pos[1] as f32, base_pos[2] as f32 + 1.0],
        5 => [base_pos[0] as f32 + 1.0, base_pos[1] as f32, base_pos[2] as f32 + 1.0],
        6 => [base_pos[0] as f32 + 1.0, base_pos[1] as f32 + 1.0, base_pos[2] as f32 + 1.0],
        7 => [base_pos[0] as f32, base_pos[1] as f32 + 1.0, base_pos[2] as f32 + 1.0],
        _ => panic!("Invalid corner index"),
    }
}

/// The 12 edges of a cube, each connecting two corners
const MC_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),  // bottom face edges
    (4, 5), (5, 6), (6, 7), (7, 4),  // top face edges
    (0, 4), (1, 5), (2, 6), (3, 7),  // vertical edges
];

/// Camera intrinsic parameters for depth projection
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    /// Focal length x
    pub fx: f32,
    /// Focal length y
    pub fy: f32,
    /// Principal point x
    pub cx: f32,
    /// Principal point y
    pub cy: f32,
}

/// Transform a 3D point using a 4x4 homogeneous matrix
fn transform_point(matrix: [[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    let x = matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3];
    let y = matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3];
    let z = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3];
    let w = matrix[3][0] * point[0] + matrix[3][1] * point[1] + matrix[3][2] * point[2] + matrix[3][3];
    [x / w, y / w, z / w]
}

/// Invert a 4x4 transformation matrix (simplified for rigid transforms)
fn invert_transform_matrix(matrix: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut inv = [[0.0; 4]; 4];
    // Rotation transpose
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = matrix[j][i];
        }
    }
    // Translation component
    let t = [matrix[0][3], matrix[1][3], matrix[2][3]];
    inv[0][3] = -inv[0][0] * t[0] - inv[0][1] * t[1] - inv[0][2] * t[2];
    inv[1][3] = -inv[1][0] * t[0] - inv[1][1] * t[1] - inv[1][2] * t[2];
    inv[2][3] = -inv[2][0] * t[0] - inv[2][1] * t[1] - inv[2][2] * t[2];
    inv[3][0] = 0.0;
    inv[3][1] = 0.0;
    inv[3][2] = 0.0;
    inv[3][3] = 1.0;
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tsdf_creation() {
        let result = TSDFVolume::new(0.01, 0.05, [10, 20, 30], [0.0, 0.0, 0.0]);
        assert!(result.is_ok());
        
        let tsdf = result.unwrap();
        assert_eq!(tsdf.dimensions(), [10, 20, 30]);
        assert_eq!(tsdf.volume_size(), 10 * 20 * 30);
        assert_eq!(tsdf.voxel_size(), 0.01);
        assert_eq!(tsdf.truncation_distance(), 0.05);
        assert_eq!(tsdf.origin(), [0.0, 0.0, 0.0]);

        // Use relative_eq for floating-point comparisons
        let physical_size = tsdf.physical_size();
        assert_relative_eq!(physical_size[0], 0.1);
        assert_relative_eq!(physical_size[1], 0.2);
        assert_relative_eq!(physical_size[2], 0.3);
        
        // Test that the volume is initialized correctly
        assert_eq!(tsdf.tsdf_data().len(), 10 * 20 * 30);
        assert_eq!(tsdf.weight_data().len(), 10 * 20 * 30);
        
        // Test that the volume is initialized to empty space
        for &value in tsdf.tsdf_data() {
            assert_eq!(value, 1.0);
        }
        
        for &value in tsdf.weight_data() {
            assert_eq!(value, 0.0);
        }
    }
    
    #[test]
    fn test_tsdf_invalid_params() {
        // Test invalid voxel size
        let result = TSDFVolume::new(0.0, 0.05, [10, 20, 30], [0.0, 0.0, 0.0]);
        assert!(result.is_err());
        
        // Test invalid truncation distance
        let result = TSDFVolume::new(0.01, 0.0, [10, 20, 30], [0.0, 0.0, 0.0]);
        assert!(result.is_err());
        
        // Test invalid dimensions
        let result = TSDFVolume::new(0.01, 0.05, [0, 20, 30], [0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_voxel_index() {
        let tsdf = TSDFVolume::new(0.01, 0.05, [10, 20, 30], [0.0, 0.0, 0.0]).unwrap();
        
        // Test corner cases
        assert_eq!(tsdf.voxel_index(0, 0, 0), 0);
        assert_eq!(tsdf.voxel_index(9, 0, 0), 9);
        assert_eq!(tsdf.voxel_index(0, 1, 0), 10);
        assert_eq!(tsdf.voxel_index(0, 0, 1), 10 * 20);
        
        // Test arbitrary position
        assert_eq!(tsdf.voxel_index(5, 10, 15), 5 + 10 * 10 + 15 * 10 * 20);
    }
    
    #[test]
    fn test_reset() {
        let mut tsdf = TSDFVolume::new(0.01, 0.05, [10, 10, 10], [0.0, 0.0, 0.0]).unwrap();
        
        // Modify the volume data
        let idx = tsdf.voxel_index(5, 5, 5);
        tsdf.tsdf_data_mut()[idx] = 0.0;
        tsdf.weight_data_mut()[idx] = 1.0;
        
        // Verify the modification
        assert_eq!(tsdf.get_tsdf(5, 5, 5), 0.0);
        assert_eq!(tsdf.get_weight(5, 5, 5), 1.0);
        
        // Reset the volume
        tsdf.reset();
        
        // Verify the reset
        assert_eq!(tsdf.get_tsdf(5, 5, 5), 1.0);
        assert_eq!(tsdf.get_weight(5, 5, 5), 0.0);
    }
    
    #[test]
    fn test_transform_point() {
        // Identity transform
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let point = [1.0, 2.0, 3.0];
        let transformed = transform_point(identity, point);
        
        assert_relative_eq!(transformed[0], point[0]);
        assert_relative_eq!(transformed[1], point[1]);
        assert_relative_eq!(transformed[2], point[2]);
        
        // Translation
        let translation = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let transformed = transform_point(translation, point);
        
        assert_relative_eq!(transformed[0], point[0] + 10.0);
        assert_relative_eq!(transformed[1], point[1] + 20.0);
        assert_relative_eq!(transformed[2], point[2] + 30.0);
    }
    
    #[test]
    fn test_invert_transform_matrix() {
        // Identity transform
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let inverted = invert_transform_matrix(identity);
        
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(inverted[i][j], identity[i][j]);
            }
        }
        
        // Translation
        let translation = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let inverted = invert_transform_matrix(translation);
        
        // Check that the rotation part is transposed (in this case, still identity)
        assert_relative_eq!(inverted[0][0], 1.0);
        assert_relative_eq!(inverted[1][1], 1.0);
        assert_relative_eq!(inverted[2][2], 1.0);
        
        // Check that the translation part is negated
        assert_relative_eq!(inverted[0][3], -10.0);
        assert_relative_eq!(inverted[1][3], -20.0);
        assert_relative_eq!(inverted[2][3], -30.0);
    }
    
    #[test]
    fn test_depth_integration() {
        // Create a small TSDF volume centered at the origin
        let mut tsdf = TSDFVolume::new(0.1, 0.2, [10, 10, 10], [-0.5, -0.5, -0.5]).unwrap();
        
        // Create a simple 3x3 depth image
        let depth_image = vec![
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
        ];
        
        // Simple camera pose
        let camera_pose = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // Simple camera intrinsics
        let intrinsics = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 1.0,
            cy: 1.0,
        };
        
        // Store the initial state of the TSDF
        let initial_tsdf = tsdf.tsdf_data().to_vec();
        let initial_weights = tsdf.weight_data().to_vec();
        
        // Integrate the depth frame
        tsdf.integrate(&depth_image, 3, 3, intrinsics, camera_pose).unwrap();
        
        // Check that the TSDF was modified
        let mut changed_tsdf = false;
        for i in 0..tsdf.volume_size() {
            if (tsdf.tsdf_data()[i] - initial_tsdf[i]).abs() > 1e-6 {
                changed_tsdf = true;
                break;
            }
        }
        assert!(changed_tsdf, "TSDF values were not modified by integration");
        
        // Check that the weights were modified
        let mut changed_weights = false;
        for i in 0..tsdf.volume_size() {
            if (tsdf.weight_data()[i] - initial_weights[i]).abs() > 1e-6 {
                changed_weights = true;
                break;
            }
        }
        assert!(changed_weights, "Weight values were not modified by integration");
        
        // Check that some voxels have positive weights
        let positive_weights_exist = tsdf.weight_data().iter().any(|&w| w > 0.0);
        assert!(positive_weights_exist, "No voxels with positive weights were found");
    }
    
    #[test]
    fn test_integrate_invalid_input() {
        let mut tsdf = TSDFVolume::new(0.01, 0.05, [10, 10, 10], [0.0, 0.0, 0.0]).unwrap();
        
        // Create a depth image with incorrect dimensions
        let depth_image = vec![1.0, 2.0, 3.0]; // only 3 pixels, not 4x4=16
        
        let camera_pose = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let intrinsics = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 2.0,
            cy: 2.0,
        };
        
        // This should fail because depth_image.len() != width * height
        let result = tsdf.integrate(&depth_image, 4, 4, intrinsics, camera_pose);
        assert!(result.is_err());
    }
} 