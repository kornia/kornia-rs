// kiss_icp.rs
// Implementation of KISS-ICP for kornia-3d
// Based on: https://github.com/PRBonn/kiss-icp
// Paper: https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/vizzo2023ral.pdf

use crate::pointcloud::PointCloud;
use std::collections::{VecDeque, HashMap};

/// A 3D vector
#[derive(Debug, Clone, Copy)]
pub struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3 {
    /// Create a new 3D vector
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Get the x component
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the y component
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Get the z component
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Get the vector as an array
    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Add another vector
    pub fn add(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }

    /// Subtract another vector
    pub fn subtract(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    }

    /// Scale the vector by a scalar
    pub fn scale(&self, scalar: f64) -> Vector3 {
        Vector3::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )
    }

    /// Compute the dot product with another vector
    pub fn dot(&self, other: &Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the cross product with another vector
    pub fn cross(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Compute the norm (length) of the vector
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize the vector (make it unit length)
    pub fn normalize(&self) -> Vector3 {
        let norm = self.norm();
        if norm > 1e-10 {
            self.scale(1.0 / norm)
        } else {
            *self
        }
    }
}

/// A 3x3 rotation matrix
#[derive(Debug, Clone)]
pub struct Matrix3 {
    data: [[f64; 3]; 3],
}

impl Matrix3 {
    /// Create a new 3x3 matrix
    pub fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    /// Create an identity matrix
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Access an element of the matrix
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    /// Set an element of the matrix
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }

    /// Multiply the matrix by a vector
    pub fn multiply_vector(&self, v: &Vector3) -> Vector3 {
        Vector3::new(
            self.data[0][0] * v.x + self.data[0][1] * v.y + self.data[0][2] * v.z,
            self.data[1][0] * v.x + self.data[1][1] * v.y + self.data[1][2] * v.z,
            self.data[2][0] * v.x + self.data[2][1] * v.y + self.data[2][2] * v.z,
        )
    }

    /// Multiply the matrix by another matrix
    pub fn multiply_matrix(&self, other: &Matrix3) -> Matrix3 {
        let mut result = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Matrix3 {
        let mut result = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }
}

/// A quaternion for representing 3D rotations
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quaternion {
    /// Create a new quaternion
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Create an identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a quaternion from axis-angle representation
    pub fn from_axis_angle(axis: &Vector3, angle: f64) -> Self {
        let half_angle = angle * 0.5;
        let s = half_angle.sin();
        let normalized_axis = axis.normalize();
        
        Self {
            w: half_angle.cos(),
            x: normalized_axis.x * s,
            y: normalized_axis.y * s,
            z: normalized_axis.z * s,
        }
    }

    /// Get the w component
    pub fn w(&self) -> f64 {
        self.w
    }

    /// Get the x component
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the y component
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Get the z component
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Normalize the quaternion
    pub fn normalize(&self) -> Quaternion {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm > 1e-10 {
            Quaternion::new(
                self.w / norm,
                self.x / norm,
                self.y / norm,
                self.z / norm,
            )
        } else {
            Quaternion::identity()
        }
    }

    /// Multiply this quaternion by another (composition of rotations)
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }

    /// Get the inverse of the quaternion
    pub fn inverse(&self) -> Quaternion {
        // For unit quaternions, the conjugate is the inverse
        let norm_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if (norm_sq - 1.0).abs() < 1e-6 {
            // If it's already normalized, just return the conjugate
            Quaternion::new(self.w, -self.x, -self.y, -self.z)
        } else {
            // Otherwise, scale the conjugate
            let scale = 1.0 / norm_sq;
            Quaternion::new(
                self.w * scale,
                -self.x * scale,
                -self.y * scale,
                -self.z * scale,
            )
        }
    }

    /// Apply the rotation to a vector
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        // Convert the vector to a quaternion with w=0
        let v_quat = Quaternion::new(0.0, v.x, v.y, v.z);
        
        // Apply the rotation: q * v * q^(-1)
        let q_inv = self.inverse();
        let rotated = self.multiply(&v_quat).multiply(&q_inv);
        
        // Extract the vector part
        Vector3::new(rotated.x, rotated.y, rotated.z)
    }

    /// Convert to a 3x3 rotation matrix
    pub fn to_matrix(&self) -> Matrix3 {
        let w = self.w;
        let x = self.x;
        let y = self.y;
        let z = self.z;
        
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;
        
        Matrix3::new([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ])
    }
}

/// A transformation in SE(3) represented by a rotation and translation
#[derive(Debug, Clone)]
pub struct SE3 {
    /// Rotation component as a quaternion
    rotation: Quaternion,
    /// Translation component as a 3D vector
    translation: Vector3,
}

impl SE3 {
    /// Create a new SE3 transformation
    pub fn new(rotation: Quaternion, translation: Vector3) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Create an identity transformation (no rotation or translation)
    pub fn identity() -> Self {
        Self {
            rotation: Quaternion::identity(),
            translation: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    /// Get the rotation component as a quaternion
    pub fn rotation(&self) -> &Quaternion {
        &self.rotation
    }

    /// Get the translation component as a 3D vector
    pub fn translation(&self) -> &Vector3 {
        &self.translation
    }

    /// Convert the transformation to a 4x4 homogeneous matrix
    pub fn as_matrix(&self) -> [[f64; 4]; 4] {
        let rot_mat = self.rotation.to_matrix();
        let trans = self.translation;
        
        [
            [rot_mat.get(0, 0), rot_mat.get(0, 1), rot_mat.get(0, 2), trans.x],
            [rot_mat.get(1, 0), rot_mat.get(1, 1), rot_mat.get(1, 2), trans.y],
            [rot_mat.get(2, 0), rot_mat.get(2, 1), rot_mat.get(2, 2), trans.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Compose this transformation with another (this * other)
    pub fn compose(&self, other: &SE3) -> SE3 {
        let new_rotation = self.rotation.multiply(&other.rotation);
        let new_translation = self.translation.add(
            &self.rotation.rotate_vector(&other.translation)
        );
        
        SE3::new(new_rotation, new_translation)
    }

    /// Get the inverse of this transformation
    pub fn inverse(&self) -> SE3 {
        let inv_rotation = self.rotation.inverse();
        let inv_translation = inv_rotation.rotate_vector(
            &Vector3::new(-self.translation.x, -self.translation.y, -self.translation.z)
        );
        
        SE3::new(inv_rotation, inv_translation)
    }
}

/// Configuration parameters for KISS-ICP
pub struct KissIcpConfig {
    /// Maximum distance threshold for point-to-plane ICP
    pub max_correspondence_distance: f64,
    /// Size of the voxel grid for downsampling the source point cloud
    pub voxel_size: f64,
    /// Maximum queue size for the local map
    pub max_points: usize,
    /// Threshold for convergence (change in transformation)
    pub convergence_threshold: f64,
    /// Maximum number of iterations for ICP
    pub max_iterations: usize,
    /// Size of the voxel grid for downsampling the local map
    pub local_map_voxel_size: f64,
}

impl Default for KissIcpConfig {
    fn default() -> Self {
        Self {
            max_correspondence_distance: 2.0,
            voxel_size: 0.5,
            max_points: 100000,
            convergence_threshold: 1e-4,
            max_iterations: 30,
            local_map_voxel_size: 1.0,
        }
    }
}

/// KISS-ICP odometry system
pub struct KissIcp {
    /// Configuration parameters
    config: KissIcpConfig,
    /// Current local map (target point cloud)
    local_map: PointCloud,
    /// Queue of recent frames to maintain the local map
    frames_queue: VecDeque<PointCloud>,
    /// Current pose estimate (trajectory)
    pose: SE3,
    /// Previous pose estimate
    prev_pose: SE3,
    /// Initial guess for the next frame
    initial_guess: SE3,
}

impl KissIcp {
    /// Create a new KISS-ICP odometry system with default parameters
    pub fn new() -> Self {
        Self::with_config(KissIcpConfig::default())
    }

    /// Create a new KISS-ICP odometry system with custom parameters
    pub fn with_config(config: KissIcpConfig) -> Self {
        Self {
            config,
            // Create empty point clouds with the proper structure
            local_map: PointCloud::new(Vec::new(), None, None),
            frames_queue: VecDeque::new(),
            pose: SE3::identity(),
            prev_pose: SE3::identity(),
            initial_guess: SE3::identity(),
        }
    }

    /// Register a new point cloud frame and update the odometry estimate
    pub fn register_frame(&mut self, frame: &PointCloud) -> SE3 {
        // Downsample the input frame
        let source = self.voxelize_frame(frame);

        // If this is the first frame, initialize the local map
        if self.local_map.is_empty() {
            self.initialize_map(&source);
            return self.pose.clone();
        }

        // Apply motion compensation to the source based on previous motion
        let compensated_source = self.apply_motion_compensation(&source);

        // Run ICP to estimate the transformation
        let transformation = self.run_icp(&compensated_source);

        // Update the pose estimate
        self.update_pose(transformation);

        // Update the local map with the new frame
        self.update_local_map(&source);

        // Return the current pose estimate
        self.pose.clone()
    }

    /// Voxelize the input point cloud based on adaptive voxel size
    fn voxelize_frame(&self, frame: &PointCloud) -> PointCloud {
        // For now, we'll implement a simple voxel grid without adaptive sizing
        // In a full implementation, this would use adaptive voxelization
        
        // Get the points from the input cloud
        let points = frame.points();
        let mut voxel_map = HashMap::new();
        
        // Simple voxel grid implementation
        for point in points.iter() {
            let voxel_x = (point[0] / self.config.voxel_size).floor() as i64;
            let voxel_y = (point[1] / self.config.voxel_size).floor() as i64;
            let voxel_z = (point[2] / self.config.voxel_size).floor() as i64;
            
            let voxel_key = (voxel_x, voxel_y, voxel_z);
            
            // Average points in the same voxel
            voxel_map.entry(voxel_key)
                .or_insert_with(|| point.clone());
        }
        
        // Create a new point cloud with voxelized points
        let voxelized_points: Vec<[f64; 3]> = voxel_map.values()
            .cloned()
            .collect();
        
        // Return a new point cloud with just the voxelized points
        PointCloud::new(voxelized_points, None, None)
    }

    /// Initialize the local map with the first frame
    fn initialize_map(&mut self, first_frame: &PointCloud) {
        // Clone the first frame to use as initial local map
        self.local_map = first_frame.clone();
        
        // Add to the frames queue
        self.frames_queue.push_back(first_frame.clone());
    }

    /// Apply motion compensation to the source point cloud based on previous motion
    fn apply_motion_compensation(&self, source: &PointCloud) -> PointCloud {
        // Get the initial guess based on constant velocity model
        // Unused currently (future scope)
        let _guess_matrix = self.initial_guess.as_matrix();
        
        // Transform the source point cloud
        let transformed_points: Vec<[f64; 3]> = source.points().iter()
            .map(|point| {
                // Convert point to Vector3
                let p = Vector3::new(point[0], point[1], point[2]);
                
                // Apply transformation
                let rotated = self.initial_guess.rotation().rotate_vector(&p);
                let transformed = rotated.add(self.initial_guess.translation());
                
                // Convert back to array
                transformed.as_array()
            })
            .collect();
        
        // Create a new point cloud with transformed points
        // Preserve normals if they exist by transforming them too
        let transformed_normals = if let Some(normals) = source.normals() {
            Some(normals.iter()
                .map(|normal| {
                    // Convert normal to Vector3
                    let n = Vector3::new(normal[0], normal[1], normal[2]);
                    
                    // Apply only rotation to normal
                    let rotated = self.initial_guess.rotation().rotate_vector(&n);
                    
                    // Convert back to array
                    rotated.as_array()
                })
                .collect())
        } else {
            None
        };
        
        // Return a new point cloud with transformed points and normals
        PointCloud::new(transformed_points, None, transformed_normals)
    }

    /// Run point-to-plane ICP to estimate the transformation
    fn run_icp(&self, source: &PointCloud) -> SE3 {
        // Implement point-to-plane ICP algorithm
        // This is a simplified version - a full implementation would be more complex
        
        // In a real implementation, you would:
        // 1. Find correspondences between source and target (local_map)
        // 2. Compute point-to-plane error metric
        // 3. Solve for the optimal transformation
        // 4. Apply transformation and iterate
        
        // For demonstration purposes, we'll implement a very basic version
        
        // Initialize transformation as identity
        let mut current_transform = SE3::identity();
        
        // Main ICP loop
        for _ in 0..self.config.max_iterations {
            // Apply current transformation to source points
            let transformed_source = self.transform_point_cloud(source, &current_transform);
            
            // Find nearest neighbors and compute point-to-plane error
            let (correspondences, has_converged) = self.find_correspondences(&transformed_source);
            
            if has_converged {
                break;
            }
            
            // Update transformation based on correspondences
            let delta_transform = self.compute_transform_update(&correspondences);
            
            // Compose transformations
            current_transform = delta_transform.compose(&current_transform);
        }
        
        current_transform
    }

    /// Transform a point cloud by an SE3 transformation
    fn transform_point_cloud(&self, cloud: &PointCloud, transform: &SE3) -> PointCloud {
        // Apply transformation to each point
        let transformed_points: Vec<[f64; 3]> = cloud.points().iter()
            .map(|point| {
                // Convert point to Vector3
                let p = Vector3::new(point[0], point[1], point[2]);
                
                // Apply transformation
                let rotated = transform.rotation().rotate_vector(&p);
                let transformed = rotated.add(transform.translation());
                
                // Convert back to array
                transformed.as_array()
            })
            .collect();
        
        // Transform normals if they exist
        let transformed_normals = if let Some(normals) = cloud.normals() {
            Some(normals.iter()
                .map(|normal| {
                    // Convert normal to Vector3
                    let n = Vector3::new(normal[0], normal[1], normal[2]);
                    
                    // Apply only rotation to normal
                    let rotated = transform.rotation().rotate_vector(&n);
                    
                    // Convert back to array
                    rotated.as_array()
                })
                .collect())
        } else {
            None
        };
        
        // Return new point cloud with transformed points and normals
        PointCloud::new(transformed_points, None, transformed_normals)
    }

    /// Find correspondences between source and target point clouds
    fn find_correspondences(&self, source: &PointCloud) -> (Vec<(usize, usize, f64)>, bool) {
        // Simplified nearest neighbor search for correspondences
        // In a real implementation, you would use a spatial data structure like KD-tree
        
        // Vector to store correspondences (source index, target index, weight)
        let mut correspondences = Vec::new();
        
        // Flag to indicate if ICP has converged
        let mut mean_distance = 0.0;
        
        // For each point in source, find closest point in target
        for (i, source_point) in source.points().iter().enumerate() {
            // Find closest point in target (brute force)
            let mut min_dist = f64::MAX;
            let mut closest_idx = 0;
            
            for (j, target_point) in self.local_map.points().iter().enumerate() {
                // Compute squared distance
                let dist = (source_point[0] - target_point[0]).powi(2) +
                          (source_point[1] - target_point[1]).powi(2) +
                          (source_point[2] - target_point[2]).powi(2);
                
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = j;
                }
            }
            
            // If distance is below threshold, add correspondence
            if min_dist.sqrt() <= self.config.max_correspondence_distance {
                correspondences.push((i, closest_idx, 1.0)); // Weight 1.0 for all points
                mean_distance += min_dist.sqrt();
            }
        }
        
        // Compute mean distance for convergence check
        if !correspondences.is_empty() {
            mean_distance /= correspondences.len() as f64;
        }
        
        // Check for convergence
        let has_converged = mean_distance < self.config.convergence_threshold;
        
        (correspondences, has_converged)
    }

    /// Compute transformation update based on point-to-plane correspondences
    /// correspondences Unused currently (future scope)
    fn compute_transform_update(&self, _correspondences: &[(usize, usize, f64)]) -> SE3 {
        // Simplified transform update
        // In a real implementation, this would solve a linear system for point-to-plane
        
        // For demonstration, return a small increment (close to identity)
        // In a real implementation, this would compute the actual transformation update
        
        // Small rotation around y-axis (for demonstration)
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let angle = 0.01; 
        let rot = Quaternion::from_axis_angle(&axis, angle);
        
        // Small translation in x direction (for demonstration)
        let trans = Vector3::new(0.01, 0.0, 0.0);
        
        SE3::new(rot, trans)
    }

    /// Update the pose estimate based on the estimated transformation
    fn update_pose(&mut self, transformation: SE3) {
        // Store the previous pose
        self.prev_pose = self.pose.clone();
        
        // Update the current pose by composing with the new transformation
        self.pose = self.pose.compose(&transformation);
        
        // Update the initial guess for the next frame based on constant velocity model
        self.initial_guess = self.prev_pose.inverse().compose(&self.pose);
    }

    /// Update the local map with the new frame
    fn update_local_map(&mut self, new_frame: &PointCloud) {
        // Transform the new frame to the global reference frame
        let transformed_frame = self.transform_point_cloud(new_frame, &self.pose);
        
        // Add the new frame to the queue
        self.frames_queue.push_back(transformed_frame);
        
        // If the queue is too large, remove the oldest frame
        if self.frames_queue.len() > self.config.max_points / new_frame.len() {
            self.frames_queue.pop_front();
        }
        
        // Rebuild the local map from the frames queue
        self.rebuild_local_map();
    }

    /// Rebuild the local map from the frames in the queue
    fn rebuild_local_map(&mut self) {
        // Combine all points from frames in the queue
        let mut all_points = Vec::new();
        
        for frame in &self.frames_queue {
            all_points.extend_from_slice(frame.points());
        }
        
        // Apply voxel grid to downsample the combined point cloud
        let mut voxel_map = HashMap::new();
        
        for point in all_points.iter() {
            let voxel_x = (point[0] / self.config.local_map_voxel_size).floor() as i64;
            let voxel_y = (point[1] / self.config.local_map_voxel_size).floor() as i64;
            let voxel_z = (point[2] / self.config.local_map_voxel_size).floor() as i64;
            
            let voxel_key = (voxel_x, voxel_y, voxel_z);
            
            // Store one point per voxel (first point encountered)
            voxel_map.entry(voxel_key)
                .or_insert_with(|| point.clone());
        }
        
        // Create a new point cloud with downsampled points
        let downsampled_points: Vec<[f64; 3]> = voxel_map.values()
            .cloned()
            .collect();
        
        // Update the local map
        self.local_map = PointCloud::new(downsampled_points, None, None);
    }

    /// Get the current pose estimate
    pub fn get_pose(&self) -> &SE3 {
        &self.pose
    }

    /// Get the current local map
    pub fn get_local_map(&self) -> &PointCloud {
        &self.local_map
    }
}

// Additional utility functions and components for the algorithm
mod utils {
    use super::*;
    
    /// Compute point-to-plane distance
    /// Unused currently (future scope)
    pub fn _point_to_plane_distance(
        point: &[f64; 3],
        plane_point: &[f64; 3],
        plane_normal: &[f64; 3]
    ) -> f64 {
        // Convert arrays to Vector3
        let p = Vector3::new(point[0], point[1], point[2]);
        let q = Vector3::new(plane_point[0], plane_point[1], plane_point[2]);
        let n = Vector3::new(plane_normal[0], plane_normal[1], plane_normal[2]);
        
        // Compute point-to-plane distance
        let v = p.subtract(&q);
        v.dot(&n).abs()
    }
    
    /// Compute the weight for a point based on its distance from sensor origin
    /// Unused currently (future scope)
    pub fn _compute_distance_weight(point: &[f64; 3], max_distance: f64) -> f64 {
        // Compute distance from origin
        let distance = (point[0].powi(2) + point[1].powi(2) + point[2].powi(2)).sqrt();
        
        // Apply weight function (e.g., linear decrease with distance)
        (1.0 - distance / max_distance).max(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kiss_icp_initialization() {
        // Create a KISS-ICP instance with default configuration
        let odometry = KissIcp::new();
        
        // Check that the pose is initialized to identity
        assert!(odometry.get_pose().translation().x().abs() < 1e-10);
        assert!(odometry.get_pose().translation().y().abs() < 1e-10);
        assert!(odometry.get_pose().translation().z().abs() < 1e-10);
        
        // Check that the local map is empty
        assert!(odometry.get_local_map().is_empty());
    }
    
    #[test]
    fn test_se3_operations() {
        // Create a rotation around Y axis
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rot = Quaternion::from_axis_angle(&axis, 0.5);
        
        // Create a translation
        let trans = Vector3::new(1.0, 0.0, 0.0);
        
        // Create an SE3 transformation
        let transform = SE3::new(rot, trans);
        
        // Check that composition with identity returns the same transform
        let identity = SE3::identity();
        let composed = transform.compose(&identity);
        
        assert!((composed.translation().x() - transform.translation().x()).abs() < 1e-10);
        assert!((composed.translation().y() - transform.translation().y()).abs() < 1e-10);
        assert!((composed.translation().z() - transform.translation().z()).abs() < 1e-10);
    }
}