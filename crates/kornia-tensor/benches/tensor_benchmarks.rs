use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kornia_tensor::{CpuAllocator, Tensor};

fn benchmark_tensor_clone(c: &mut Criterion) {
    let tensor =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();

    c.bench_function("tensor clone", |b| {
        b.iter(|| {
            let _clone = black_box(&tensor).clone();
        })
    });
}

fn benchmark_element_wise_op(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();

    c.bench_function("element_wise_op", |b| {
        b.iter(|| {
            let _result = black_box(&tensor1)
                .element_wise_op(black_box(&tensor2), |a, b| *a + *b)
                .unwrap();
        })
    });
}

fn benchmark_element_wise_op_inplace(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();

    c.bench_function("element_wise_op_inplace", |b| {
        b.iter(|| {
            let mut t = black_box(&tensor1).clone();
            t.element_wise_op_inplace(black_box(&tensor2), |a, b| *a += *b)
                .unwrap();
        })
    });
}

fn benchmark_add(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();

    c.bench_function("add", |b| {
        b.iter(|| {
            let _result = kornia_tensor::add(black_box(&tensor1), black_box(&tensor2)).unwrap();
        })
    });
}

fn benchmark_add_inplace(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();

    c.bench_function("add_inplace", |b| {
        b.iter(|| {
            let mut t = black_box(&tensor1).clone();
            kornia_tensor::add_inplace(&mut t, black_box(&tensor2)).unwrap();
        })
    });
}

fn benchmark_chain_ops(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();
    let tensor3 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![3.0; 10000], CpuAllocator).unwrap();

    c.bench_function("chain_ops", |b| {
        b.iter(|| {
            let t1 = black_box(&tensor1)
                .element_wise_op(black_box(&tensor2), |a, b| *a + *b)
                .unwrap();
            let t2 = t1
                .element_wise_op(black_box(&tensor3), |a, b| *a * *b)
                .unwrap();
            let _t3 = t2
                .element_wise_op(black_box(&tensor1), |a, b| *a - *b)
                .unwrap();
        })
    });
}

fn benchmark_chain_ops_inplace(c: &mut Criterion) {
    let tensor1 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![1.0; 10000], CpuAllocator).unwrap();
    let tensor2 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![2.0; 10000], CpuAllocator).unwrap();
    let tensor3 =
        Tensor::<f32, 2, _>::from_shape_vec([100, 100], vec![3.0; 10000], CpuAllocator).unwrap();

    c.bench_function("chain_ops_inplace", |b| {
        b.iter(|| {
            let mut t = black_box(&tensor1).clone();
            kornia_tensor::add_inplace(&mut t, black_box(&tensor2)).unwrap();
            kornia_tensor::mul_inplace(&mut t, black_box(&tensor3)).unwrap();
            kornia_tensor::sub_inplace(&mut t, black_box(&tensor1)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    benchmark_tensor_clone,
    benchmark_element_wise_op,
    benchmark_element_wise_op_inplace,
    benchmark_add,
    benchmark_add_inplace,
    benchmark_chain_ops,
    benchmark_chain_ops_inplace
);
criterion_main!(benches);
