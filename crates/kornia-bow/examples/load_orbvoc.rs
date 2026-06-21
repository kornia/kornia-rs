use std::time::Instant;

use kornia_bow::orb_slam3::{load_orb_slam3_vocabulary, pack_orb_descriptor};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: load_orbvoc <path-to-ORBvoc.txt>");

    println!("loading vocabulary from {}", path);
    let t0 = Instant::now();
    let vocab = load_orb_slam3_vocabulary(&path).expect("failed to load vocabulary");
    let dt = t0.elapsed();
    println!(
        "loaded {} blocks in {:.2}s",
        vocab.blocks.len(),
        dt.as_secs_f32()
    );

    // Transform a zero descriptor and a random one to verify traversal works.
    let zero = pack_orb_descriptor(&[0u8; 32]);
    let (word, weight) = vocab.transform_one(&zero);
    println!("zero-desc → word_id={}, idf_weight={}", word, weight);

    let mut pattern = [0u8; 32];
    for (i, b) in pattern.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17);
    }
    let p = pack_orb_descriptor(&pattern);
    let (word, weight) = vocab.transform_one(&p);
    println!("pattern-desc → word_id={}, idf_weight={}", word, weight);
}
