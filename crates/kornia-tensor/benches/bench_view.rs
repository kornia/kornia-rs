use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor::{CpuAllocator, Tensor};

fn sample_tensor() -> Tensor<u8, 3, CpuAllocator> {
    Tensor::from_shape_val([1080, 1080, 3], 0_u8, CpuAllocator)
}

fn bench_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("View");

    group.bench_function("as_contiguous", |b| {
        b.iter_batched(
            sample_tensor,
            |tv| black_box(tv).permute_axes([2, 0, 1]).as_contiguous(),
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_image);
criterion_main!(benches);
