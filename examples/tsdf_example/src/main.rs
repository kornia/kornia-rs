use argh::FromArgs;
use kornia_3d::tsdf::{TSDFVolume, CameraIntrinsics};

fn default_dimensions() -> String {
    String::from("[100, 100, 100]")
}

fn default_origin() -> String {
    String::from("[-0.5, -0.5, 0.0]")
}

#[derive(FromArgs)]
/// TSDF Volume Integration Example
struct Args {
    /// voxel size in meters
    #[argh(option, short = 'v', default = "0.01")]
    voxel_size: f32,

    /// truncation distance in meters
    #[argh(option, short = 't', default = "0.05")]
    truncation_distance: f32,

    /// volume dimensions in voxels (x,y,z)
    #[argh(option, short = 'd', default = "default_dimensions()")]
    dimensions: String,

    /// volume origin in world coordinates (x,y,z)
    #[argh(option, short = 'o', default = "default_origin()")]
    origin: String,
}

fn parse_array<T: std::str::FromStr + Clone>(s: &str) -> Result<[T; 3], String>
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let s = s.trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    
    if parts.len() != 3 {
        return Err(format!("Expected 3 values, got {}", parts.len()));
    }
    
    let mut result = Vec::with_capacity(3);
    for part in parts {
        match part.parse::<T>() {
            Ok(value) => result.push(value),
            Err(e) => return Err(format!("Failed to parse value '{}': {}", part, e)),
        }
    }
    
    Ok([result[0].clone(), result[1].clone(), result[2].clone()])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    
    // Parse dimensions and origin
    let dimensions = parse_array::<usize>(&args.dimensions)?;
    let origin = parse_array::<f32>(&args.origin)?;
    
    println!("Creating TSDF volume...");
    println!("  Voxel size: {}", args.voxel_size);
    println!("  Truncation distance: {}", args.truncation_distance);
    println!("  Dimensions: [{}, {}, {}]", dimensions[0], dimensions[1], dimensions[2]);
    println!("  Origin: [{}, {}, {}]", origin[0], origin[1], origin[2]);
    
    // Create a TSDF volume
    let mut tsdf = TSDFVolume::new(
        args.voxel_size,
        args.truncation_distance,
        dimensions,
        origin,
    )?;
    
    // Create a synthetic depth image representing a plane at z=0
    let width = 320;
    let height = 240;
    let mut depth_image = vec![0.0f32; width * height];
    
    // Fill with a simple depth pattern - a plane at z=0
    for y in 0..height {
        for x in 0..width {
            // Distance from camera at (0,0,1) to point on z=0 plane
            depth_image[y * width + x] = 1.0;
        }
    }
    
    // Camera intrinsics (typical for a 320x240 depth camera)
    let intrinsics = CameraIntrinsics {
        fx: 240.0, // focal length x
        fy: 240.0, // focal length y
        cx: 160.0, // principal point x
        cy: 120.0, // principal point y
    };
    
    // Camera poses for different viewpoints
    let camera_poses = [
        // Front view
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        // Side view
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        // Top view
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ];
    
    // Integrate multiple frames from different viewpoints
    for (i, &pose) in camera_poses.iter().enumerate() {
        println!("Integrating frame {} of {}", i + 1, camera_poses.len());
        tsdf.integrate(&depth_image, width, height, intrinsics, pose)?;
    }
    
    // Extract the mesh from the TSDF volume
    println!("Extracting mesh...");
    let (vertices, triangles) = tsdf.extract_mesh()?;
    println!("  Extracted {} vertices and {} triangles", vertices.len(), triangles.len());
    
    // Print some metrics about the volume
    let z_mid = dimensions[2] / 2;
    let mut negative_count = 0;
    let mut positive_count = 0;
    let mut zero_count = 0;
    
    for y in 0..dimensions[1] {
        for x in 0..dimensions[0] {
            let value = tsdf.get_tsdf(x, y, z_mid);
            if value < -0.01 {
                negative_count += 1;
            } else if value > 0.01 {
                positive_count += 1;
            } else {
                zero_count += 1;
            }
        }
    }
    
    println!("TSDF volume metrics at middle slice (z={}):", z_mid);
    println!("  Negative values (inside object): {}", negative_count);
    println!("  Near-zero values (on surface): {}", zero_count);
    println!("  Positive values (outside object): {}", positive_count);
    
    println!("TSDF integration completed successfully.");
    
    Ok(())
} 