use criterion::{criterion_group, criterion_main, Criterion};
use kornia_bow::{
    metric::{Feature, Hamming},
    BlockCluster, Vocabulary,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::mem;

const B: usize = 10;
const D: usize = 4;

fn generate_random_descriptors(count: usize) -> Vec<Feature<u64, D>> {
    let mut rng = StdRng::from_seed([0; 32]);
    (0..count)
        .map(|_| {
            let mut descriptor = [0; D];
            for val in descriptor.iter_mut() {
                *val = rng.random();
            }
            Feature(descriptor)
        })
        .collect()
}

fn bench_transform_one(c: &mut Criterion) {
    // Generate training data
    let training_data = generate_random_descriptors(1000);

    // Train the vocabulary directly
    let vocab = Vocabulary::<B, Hamming<D>>::train(&training_data, 3).unwrap();

    // --- Memory Stats ---
    let blocks_size = vocab.blocks.capacity() * mem::size_of::<BlockCluster<B, Hamming<D>>>();
    println!("\n--- Vocabulary Memory Stats ---");
    println!(
        "BlockClusters: {} bytes (Count: {})",
        blocks_size,
        vocab.blocks.len()
    );
    println!(
        "Total Size:    {} bytes ({:.2} MB)",
        blocks_size,
        blocks_size as f64 / 1024.0 / 1024.0
    );
    println!("-------------------------------\n");

    // Generate test data
    let test_data = generate_random_descriptors(1)[0];

    c.bench_function("transform_one", |b| {
        b.iter(|| {
            vocab.transform_one(&test_data);
        })
    });
}

fn bench_train(c: &mut Criterion) {
    let training_data = generate_random_descriptors(2000);

    c.bench_function("train_vocabulary", |b| {
        b.iter(|| {
            Vocabulary::<B, Hamming<D>>::train(&training_data, 3).unwrap();
        })
    });
}

fn bench_transform_batch(c: &mut Criterion) {
    let training_data = generate_random_descriptors(1000);
    let vocab = Vocabulary::<B, Hamming<D>>::train(&training_data, 3).unwrap();
    let test_features_100 = generate_random_descriptors(100);
    let test_features_1000 = generate_random_descriptors(1000);

    c.bench_function("transform_batch_100", |b| {
        b.iter(|| {
            vocab.transform(&test_features_100).unwrap();
        })
    });

    c.bench_function("transform_batch_1000", |b| {
        b.iter(|| {
            vocab.transform(&test_features_1000).unwrap();
        })
    });
}

fn bench_transform_direct_index(c: &mut Criterion) {
    let training_data = generate_random_descriptors(1000);
    let vocab = Vocabulary::<B, Hamming<D>>::train(&training_data, 3).unwrap();
    let test_features_100 = generate_random_descriptors(100);
    let test_features_1000 = generate_random_descriptors(1000);

    c.bench_function("transform_direct_index_100", |b| {
        b.iter(|| {
            vocab
                .transform_with_direct_index(&test_features_100, 1)
                .unwrap();
        })
    });

    c.bench_function("transform_direct_index_1000", |b| {
        b.iter(|| {
            vocab
                .transform_with_direct_index(&test_features_1000, 1)
                .unwrap();
        })
    });
}

fn bench_bow_l1(c: &mut Criterion) {
    let training_data = generate_random_descriptors(1000);
    let vocab = Vocabulary::<B, Hamming<D>>::train(&training_data, 3).unwrap();

    let feat1 = generate_random_descriptors(100);
    let feat2 = generate_random_descriptors(100);

    let bow1 = vocab.transform(&feat1).unwrap();
    let bow2 = vocab.transform(&feat2).unwrap();

    c.bench_function("bow_l1_similarity", |b| {
        b.iter(|| {
            bow1.l1_similarity(&bow2);
        })
    });
}

criterion_group!(
    benches,
    bench_transform_one,
    bench_train,
    bench_transform_batch,
    bench_transform_direct_index,
    bench_bow_l1
);
criterion_main!(benches);
