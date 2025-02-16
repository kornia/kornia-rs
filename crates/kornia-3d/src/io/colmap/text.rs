use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use super::{CameraModelId, ColmapCamera, ColmapImage, ColmapPoint3d};

/// Error types for the COLMAP module.
#[derive(Debug, thiserror::Error)]
pub enum ColmapError {
    /// Error reading or writing file
    #[error("error reading or writing file")]
    IoError(#[from] std::io::Error),

    /// Invalid number of camera parameters
    #[error("Invalid number of camera parameters")]
    InvalidNumCameraParams(usize),

    /// Parse error
    #[error("Parse error {0}")]
    ParseError(String),
}

/// Read the cameras.txt file and return a vector of ColmapCamera structs.
///
/// # Arguments
///
/// * `path` - The path to the cameras.txt file.
///
/// # Returns
///
/// A vector of ColmapCamera structs.
pub fn read_cameras_txt(path: impl AsRef<Path>) -> Result<Vec<ColmapCamera>, ColmapError> {
    // open the file and create a buffered reader
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // skip the first 3 lines containing the header and parse the rest
    let cameras = reader
        .lines()
        .skip(3)
        .map(|line| -> Result<ColmapCamera, ColmapError> {
            let line = line.map_err(ColmapError::from)?;
            parse_camera_line(&line)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(cameras)
}

/// Read the points3D.txt file and return a vector of ColmapPoint3d structs.
///
/// # Arguments
///
/// * `path` - The path to the points3D.txt file.
///
/// # Returns
///
/// A vector of ColmapPoint3d structs.
pub fn read_points3d_txt(path: impl AsRef<Path>) -> Result<Vec<ColmapPoint3d>, ColmapError> {
    // open the file and create a buffered reader
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // skip the first 3 lines containing the header and parse the rest

    let points = reader
        .lines()
        .skip(3)
        .map(|line| -> Result<ColmapPoint3d, ColmapError> {
            let line = line.map_err(ColmapError::from)?;
            parse_point3d_line(&line)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(points)
}

/// Read the images.txt file and return a vector of ColmapImage structs.
///
/// # Arguments
///
/// * `path` - The path to the images.txt file.
///
/// # Returns
///
/// A vector of ColmapImage structs.
pub fn read_images_txt(path: impl AsRef<Path>) -> Result<Vec<ColmapImage>, ColmapError> {
    // open the file and create a buffered reader
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let images = reader
        .lines()
        .skip(4)
        .collect::<Result<Vec<_>, _>>()?
        .chunks(2)
        .map(|chunk| match chunk {
            [line1, line2] => parse_image_line(line1, line2),
            _ => Err(ColmapError::ParseError(
                "Invalid number of lines".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(images)
}

/// Utility functions for parsing COLMAP text files
fn parse_part<T: std::str::FromStr>(s: &str) -> Result<T, ColmapError>
where
    T::Err: std::fmt::Display,
{
    s.parse::<T>()
        .map_err(|e| ColmapError::ParseError(format!("{}: {}", s, e)))
}

/// Parse a camera line and return a ColmapCamera struct.
/// NOTE: The number of parameters depends on the camera model.
///       CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[0], PARAMS[1], ...
fn parse_camera_line(line: &str) -> Result<ColmapCamera, ColmapError> {
    // split the line into parts by whitespace
    let parts = line.split_whitespace().collect::<Vec<_>>();

    if parts.len() < 5 {
        return Err(ColmapError::ParseError(format!(
            "Invalid number of parts: {}",
            parts.len()
        )));
    }

    Ok(ColmapCamera {
        camera_id: parse_part(parts[0])?,
        model_id: parse_camera_model_id(parts[1])?,
        width: parse_part(parts[2])?,
        height: parse_part(parts[3])?,
        params: parts[4..]
            .iter()
            .map(|s| parse_part(s))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn parse_camera_model_id(model_id: &str) -> Result<CameraModelId, ColmapError> {
    match model_id {
        "SIMPLE_PINHOLE" => Ok(CameraModelId::CameraModelSimplePinhole),
        "PINHOLE" => Ok(CameraModelId::CameraModelPinhole),
        "SIMPLE_RADIAL" => Ok(CameraModelId::CameraModelSimplifiedRadial),
        "RADIAL" => Ok(CameraModelId::CameraModelRadial),
        "OPENCV" => Ok(CameraModelId::CameraModelOpenCV),
        "OPENCV_FISHEYE" => Ok(CameraModelId::CameraModelOpenCVFisheye),
        "FULL_OPENCV" => Ok(CameraModelId::CameraModelFullOpenCV),
        "FOV" => Ok(CameraModelId::CameraModelFOV),
        "SIMPLE_RADIAL_FISHEYE" => Ok(CameraModelId::CameraModelSimpleRadialFisheye),
        "RADIAL_FISHEYE" => Ok(CameraModelId::CameraModelRadialFisheye),
        "THIN_PRISM_FISHEYE" => Ok(CameraModelId::CameraModelThinPrismFisheye),
        _ => Err(ColmapError::ParseError(format!(
            "Invalid camera model id: {}",
            model_id
        ))),
    }
}

/// Parse a point3d line and return a ColmapPoint3d struct.
/// NOTE: The number of parameters depends on the camera model.
///       POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[0], TRACK[1], ...
fn parse_point3d_line(line: &str) -> Result<ColmapPoint3d, ColmapError> {
    // split the line into parts by whitespace
    let parts = line.split_whitespace().collect::<Vec<_>>();

    // check if the number of parts is correct
    if parts.len() < 8 {
        return Err(ColmapError::ParseError(format!(
            "Invalid number of parts: {}",
            parts.len()
        )));
    }

    Ok(ColmapPoint3d {
        point3d_id: parse_part(parts[0])?,
        xyz: parts[1..4]
            .iter()
            .map(|s| parse_part(s))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                ColmapError::ParseError("Invalid number of xyz coordinates".to_string())
            })?,
        rgb: parts[4..7]
            .iter()
            .map(|s| parse_part(s))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                ColmapError::ParseError("Invalid number of rgb coordinates".to_string())
            })?,
        error: parse_part(parts[7])?,
        track: parts[8..]
            .chunks_exact(2)
            .map(|chunk| -> Result<(u32, u32), ColmapError> {
                Ok((parse_part(chunk[0])?, parse_part(chunk[1])?))
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

/// Parse an image line and return a ColmapImage struct.
/// #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
/// #   POINTS2D[] as (X, Y, POINT3D_ID)
fn parse_image_line(line1: &str, line2: &str) -> Result<ColmapImage, ColmapError> {
    // split the line into parts by whitespace
    let parts1 = line1.split_whitespace().collect::<Vec<_>>();
    let parts2 = line2.split_whitespace().collect::<Vec<_>>();

    Ok(ColmapImage {
        image_id: parse_part(parts1[0])?,
        rotation: parts1[1..5]
            .iter()
            .map(|s| parse_part(s))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                ColmapError::ParseError("Invalid number of rotation coordinates".to_string())
            })?,
        translation: parts1[5..8]
            .iter()
            .map(|s| parse_part(s))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                ColmapError::ParseError("Invalid number of translation coordinates".to_string())
            })?,
        camera_id: parse_part(parts1[8])?,
        name: parts1[9].to_string(),
        points2d: parts2
            .chunks_exact(3)
            .map(|chunk| -> Result<(f64, f64, i64), ColmapError> {
                Ok((
                    parse_part(chunk[0])?,
                    parse_part(chunk[1])?,
                    parse_part(chunk[2])?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}
