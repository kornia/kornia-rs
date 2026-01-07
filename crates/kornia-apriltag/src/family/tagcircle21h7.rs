use super::*;

impl TagFamily {
    /// The TagCircle21H7 AprilTag Family.
    pub fn tagcircle21_h7() -> Self {
        Self {
            name: "tagcircle21_h7".to_string(),
            width_at_border: 5,
            reversed_border: true,
            total_width: 9,
            nbits: 21,
            bit_x: vec![
                1, 2, 3, 1, 2, 6, 6, 6, 3, 3, 3, 2, 1, 3, 2, -2, -2, -2, 1, 1, 2,
            ],
            bit_y: vec![
                -2, -2, -2, 1, 1, 1, 2, 3, 1, 2, 6, 6, 6, 3, 3, 3, 2, 1, 3, 2, 2,
            ],
            code_data: CODE_DATA.into(),
            quick_decode: QuickDecode::new(21, &CODE_DATA, HammingConfig::default()),
            sharpening_buffer: SharpeningBuffer::new(81),
        }
    }
}

pub static CODE_DATA: [usize; 38] = [
    0x0000000000157863,
    0x0000000000047e28,
    0x00000000001383ed,
    0x000000000000953c,
    0x00000000000da68b,
    0x00000000001cac50,
    0x00000000000bb215,
    0x000000000016ceee,
    0x000000000005d4b3,
    0x00000000001ff751,
    0x00000000000efd16,
    0x0000000000072b3e,
    0x0000000000163103,
    0x0000000000106e56,
    0x00000000001996b9,
    0x00000000000c0234,
    0x00000000000624d2,
    0x00000000001fa985,
    0x00000000000344a5,
    0x00000000000762fb,
    0x000000000019e92b,
    0x0000000000043755,
    0x000000000001a4f4,
    0x000000000010fad8,
    0x0000000000001b52,
    0x000000000017e59f,
    0x00000000000e6f70,
    0x00000000000ed47a,
    0x00000000000c9931,
    0x0000000000014df2,
    0x00000000000a06f1,
    0x00000000000e5041,
    0x000000000012ec03,
    0x000000000016724e,
    0x00000000000af1a5,
    0x000000000008a8ac,
    0x0000000000015b39,
    0x00000000001ec1e3,
];
