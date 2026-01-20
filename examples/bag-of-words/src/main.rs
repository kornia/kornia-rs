use kornia_bow::metric::{Feature, Hamming};
use kornia_bow::Vocabulary;
use rand::{rngs::StdRng, Rng, SeedableRng};

const BRANCHING_FACTOR: usize = 10;
const DESC_DIM: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== kornia-bow Feature Demo ===\n");

    // Generate dummy data
    println!("-> Generating descriptors...");
    let mut rng = StdRng::seed_from_u64(42);

    let train_data: Vec<Feature<u64, DESC_DIM>> = (0..10_000)
        .map(|_| {
            let mut desc = [0u64; DESC_DIM];
            rng.fill(&mut desc);
            Feature(desc)
        })
        .collect();

    let query1: Vec<Feature<u64, DESC_DIM>> = (0..500)
        .map(|_| {
            let mut desc = [0u64; DESC_DIM];
            rng.fill(&mut desc);
            Feature(desc)
        })
        .collect();

    let query2: Vec<Feature<u64, DESC_DIM>> = query1
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut desc = f.0;
            if i % 10 == 0 {
                desc[0] ^= 1;
            }
            Feature(desc)
        })
        .collect();

    // Train
    println!("-> Training vocabulary...");
    let vocab = Vocabulary::<BRANCHING_FACTOR, Hamming<DESC_DIM>>::train(&train_data, 3)?;

    // Persistence
    println!("-> Saving and loading...");
    let path = "temp_vocab.bin";
    vocab.save(path)?;
    let loaded_vocab = Vocabulary::<BRANCHING_FACTOR, Hamming<DESC_DIM>>::load(path)?;
    std::fs::remove_file(path)?;

    // Transform
    println!("-> Transforming to BoW vectors...");
    let mut bow1 = loaded_vocab.transform(&query1)?;
    let mut bow2 = loaded_vocab.transform(&query2)?;

    // Similarity scores (L1)
    println!("-> Similarity Scores (L1-normalized):");
    bow1.normalize_l1();
    bow2.normalize_l1();

    let s_l1 = bow1.l1_similarity(&bow2);
    let s_chi = bow1.chi_square(&bow2);
    let s_kl = bow1.kl_divergence(&bow2);
    let s_bhat = bow1.bhattacharyya(&bow2);

    println!("   L1 Score:      {:.4}", s_l1);
    println!("   Bhattacharyya: {:.4}", s_bhat);
    println!("   Chi-Square:    {:.4}", s_chi);
    println!("   KL Divergence: {:.4}", s_kl);

    // Similarity scores (L2)
    println!("\n-> Similarity Scores (L2-normalized):");
    let mut bow1_l2 = bow1.clone();
    let mut bow2_l2 = bow2.clone();
    bow1_l2.normalize_l2();
    bow2_l2.normalize_l2();

    println!("   L2 Score:      {:.4}", bow1_l2.l2(&bow2_l2));
    println!("   Dot Product:   {:.4}", bow1_l2.dot_product(&bow2_l2));

    // Direct Index
    println!("\n-> Direct Index (level 2):");
    let (_bow, direct_index) = loaded_vocab.transform_with_direct_index(&query1, 2)?;
    println!("   Nodes at level 2: {}", direct_index.0.len());

    println!("\n=== Demo Complete ===");
    Ok(())
}
