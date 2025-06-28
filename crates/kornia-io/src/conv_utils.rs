/// Utility function to convert 16-bit `Vec<u8>` to `Vec<u16>`
pub fn convert_buf_u8_u16(buf: Vec<u8>) -> Vec<u16> {
    let mut buf_u16 = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        buf_u16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }

    buf_u16
}

// This function expects the size of output to be input.len() / 2;
pub fn convert_buf_u8_u16_into_slice(input: &[u8], output: &mut [u16]) {
    for (i, chunk) in input.chunks_exact(2).enumerate() {
        output[i] = u16::from_be_bytes([chunk[0], chunk[1]]);
    }
}

pub fn convert_buf_u16_u8(buf: &[u16]) -> Vec<u8> {
    let mut buf_u8: Vec<u8> = Vec::with_capacity(buf.len() * 2);

    for byte in buf {
        let be_bytes = byte.to_be_bytes();
        buf_u8.extend_from_slice(&be_bytes);
    }

    buf_u8
}
