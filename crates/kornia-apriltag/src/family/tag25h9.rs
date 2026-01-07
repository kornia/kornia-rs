use super::*;

impl TagFamilyBuilder {
    /// Creates a builder for the Tag25H9 AprilTag family.
    pub fn tag25_h9() -> Self {
        Self::new(
            "tag25_h9",
            7,
            false,
            9,
            25,
            vec![
                1, 2, 3, 4, 2, 3, 5, 5, 5, 5, 4, 4, 5, 4, 3, 2, 4, 3, 1, 1, 1, 1, 2, 2, 3,
            ],
            vec![
                1, 1, 1, 1, 2, 2, 1, 2, 3, 4, 2, 3, 5, 5, 5, 5, 4, 4, 5, 4, 3, 2, 4, 3, 3,
            ],
            CODE_DATA.into(),
        )
    }
}

impl TagFamily {
    /// The Tag25H9 AprilTag family with default configuration.
    pub fn tag25_h9() -> Self {
        TagFamilyBuilder::tag25_h9().build()
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
