use super::*;
use crate::errors::AprilTagError;

impl TagFamily {
    /// The Tag16H5 AprilTag family.
    pub fn tag16_h5() -> Result<Self, AprilTagError> {
        Ok(Self {
            name: "tag16_h5".to_string(),
            width_at_border: 6,
            reversed_border: false,
            total_width: 8,
            nbits: 16,
            bit_x: vec![1, 2, 3, 2, 4, 4, 4, 3, 4, 3, 2, 3, 1, 1, 1, 2],
            bit_y: vec![1, 1, 1, 2, 1, 2, 3, 2, 4, 4, 4, 3, 4, 3, 2, 3],
            code_data: CODE_DATA.into(),
            quick_decode: QuickDecode::new(16, &CODE_DATA, 2)?,
            sharpening_buffer: SharpeningBuffer::new(64),
        })
    }
}

pub static CODE_DATA: [usize; 30] = [
    0x00000000000027c8,
    0x00000000000031b6,
    0x0000000000003859,
    0x000000000000569c,
    0x0000000000006c76,
    0x0000000000007ddb,
    0x000000000000af09,
    0x000000000000f5a1,
    0x000000000000fb8b,
    0x0000000000001cb9,
    0x00000000000028ca,
    0x000000000000e8dc,
    0x0000000000001426,
    0x0000000000005770,
    0x0000000000009253,
    0x000000000000b702,
    0x000000000000063a,
    0x0000000000008f34,
    0x000000000000b4c0,
    0x00000000000051ec,
    0x000000000000e6f0,
    0x0000000000005fa4,
    0x000000000000dd43,
    0x0000000000001aaa,
    0x000000000000e62f,
    0x0000000000006dbc,
    0x000000000000b6eb,
    0x000000000000de10,
    0x000000000000154d,
    0x000000000000b57a,
];
