//! Converts an ORB-SLAM3 text vocabulary (`ORBvoc.txt`) into kornia-bow's
//! bincode format for fast runtime loading.
//!
//! Usage: `convert_orbvoc <ORBvoc.txt> <out.bin>`

use std::time::Instant;

use kornia_bow::orb_slam3::load_orb_slam3_vocabulary;

fn main() {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .expect("usage: convert_orbvoc <ORBvoc.txt> <out.bin>");
    let dst = args
        .next()
        .expect("usage: convert_orbvoc <ORBvoc.txt> <out.bin>");

    let t0 = Instant::now();
    let vocab = load_orb_slam3_vocabulary(&src).expect("failed to parse text vocabulary");
    println!(
        "parsed {} blocks from {} in {:.2}s",
        vocab.blocks.len(),
        src,
        t0.elapsed().as_secs_f32()
    );

    let t1 = Instant::now();
    vocab.save(&dst).expect("failed to save bincode vocabulary");
    println!("wrote {} in {:.2}s", dst, t1.elapsed().as_secs_f32());
}
