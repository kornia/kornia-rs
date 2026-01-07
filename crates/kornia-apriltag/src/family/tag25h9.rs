use super::*;

impl TagFamily {
    /// The Tag25H9 AprilTag Family.
    pub fn tag25_h9() -> Self {
        Self {
            name: "tag25_h9".to_string(),
            width_at_border: 7,
            reversed_border: false,
            total_width: 9,
            nbits: 25,
            bit_x: vec![
                1, 2, 3, 4, 2, 3, 5, 5, 5, 5, 4, 4, 5, 4, 3, 2, 4, 3, 1, 1, 1, 1, 2, 2, 3,
            ],
            bit_y: vec![
                1, 1, 1, 1, 2, 2, 1, 2, 3, 4, 2, 3, 5, 5, 5, 5, 4, 4, 5, 4, 3, 2, 4, 3, 3,
            ],
            code_data: CODE_DATA.into(),
            quick_decode: QuickDecode::new(25, &CODE_DATA, HammingConfig::default()),
            sharpening_buffer: SharpeningBuffer::new(81),
        }
    }
}

pub static CODE_DATA: [usize; 35] = [
    0x000000000156f1f4,
    0x0000000001f28cd5,
    0x00000000016ce32c,
    0x0000000001ea379c,
    0x0000000001390f89,
    0x000000000034fad0,
    0x00000000007dcdb5,
    0x000000000119ba95,
    0x0000000001ae9daa,
    0x0000000000df02aa,
    0x000000000082fc15,
    0x0000000000465123,
    0x0000000000ceee98,
    0x0000000001f17260,
    0x00000000014429cd,
    0x00000000017248a8,
    0x00000000016ad452,
    0x00000000009670ad,
    0x00000000016f65b2,
    0x0000000000b8322b,
    0x00000000005d715b,
    0x0000000001a1c7e7,
    0x0000000000d7890d,
    0x0000000001813522,
    0x0000000001c9c611,
    0x000000000099e4a4,
    0x0000000000855234,
    0x00000000017b81c0,
    0x0000000000c294bb,
    0x000000000089fae3,
    0x000000000044df5f,
    0x0000000001360159,
    0x0000000000ec31e8,
    0x0000000001bcc0f6,
    0x0000000000a64f8d,
];
