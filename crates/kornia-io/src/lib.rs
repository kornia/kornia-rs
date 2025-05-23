#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Module to handle the error types for the io module.
pub mod error;

/// Module to handle the camera frame rate.
pub mod fps_counter;

/// High-level read and write functions for images.
pub mod functional;

/// TurboJPEG image encoding and decoding.
#[cfg(feature = "turbojpeg")]
pub mod jpegturbo;

/// PNG image encoding and decoding.
pub mod png;

/// JPEG image encoding and decoding.
pub mod jpeg;

/// GStreamer video module for real-time video processing.
#[cfg(feature = "gstreamer")]
pub mod stream;

/// TIFF image encoding and decoding.
pub mod tiff;

pub use crate::error::IoError;

/// Utility function to convert 16-bit `Vec<u8>` to `Vec<u16>`
pub(crate) fn convert_buf_u8_u16(buf: Vec<u8>) -> Vec<u16> {
    let mut buf_u16 = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        buf_u16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }

    buf_u16
}

// This function expects the size of output to be input.len() / 2;
pub(crate) fn convert_buf_u8_u16_into_slice(input: &[u8], output: &mut [u16]) {
    for (i, chunk) in input.chunks_exact(2).enumerate() {
        output[i] = u16::from_be_bytes([chunk[0], chunk[1]]);
    }
}

pub(crate) fn convert_buf_u16_u8(buf: &[u16]) -> Vec<u8> {
    let mut buf_u8: Vec<u8> = Vec::with_capacity(buf.len() * 2);

    for byte in buf {
        let be_bytes = byte.to_be_bytes();
        buf_u8.extend_from_slice(&be_bytes);
    }

    buf_u8
}
