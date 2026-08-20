//! CPU vs GPU benchmark for `kornia_tensor::ops`.
//!
//! Reports three numbers per case, following the same H2D / kernel / D2H
//! split the `kornia-imgproc` CUDA benches use:
//!
//! * **CPU** — the host path (`MemoryDomain::Host`), which is the scalar
//!   single-threaded loop in `ops.rs`.
//! * **GPU kernel** — the op on tensors that are *already* device-resident,
//!   with the stream synchronized inside the timed region. This is the number
//!   that matters for a pipeline keeping data on the device across ops.
//! * **GPU round-trip** — upload, op, download. This is what a caller pays
//!   when the data starts and ends on the host.
//!
//! ```sh
//! cargo bench -p kornia-tensor --bench bench_tensor_ops --features cuda
//! ```
//!
//! Sizes follow the 1M / 10M / 100M element sweep; 100M f32 is 400 MB per
//! operand, so the binary cases allocate 1.2 GB of device memory and are
//! skipped automatically when the allocation fails.

fn main() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("bench_tensor_ops requires --features cuda");

    #[cfg(feature = "cuda")]
    cuda_bench::run();
}

#[cfg(feature = "cuda")]
mod cuda_bench {
    use std::time::Instant;

    use cudarc::driver::CudaContext;
    use kornia_tensor::{
        ops::{self, BinaryOp, ReduceOp, UnaryOp},
        Tensor,
    };

    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    /// Element counts: 1M, 10M, 100M.
    const SIZES: &[(&str, usize)] = &[
        ("1M", 1_000_000),
        ("10M", 10_000_000),
        ("100M", 100_000_000),
    ];

    /// Minimum per-iteration wall time, in milliseconds.
    ///
    /// The minimum rather than the mean: the host path is a plain scalar loop
    /// and is highly sensitive to scheduler noise on a loaded machine (observed
    /// 2-6x spread across runs), while the GPU kernel times are stable to
    /// within a few percent. The fastest observed iteration is the one least
    /// contaminated by unrelated work, and taking it for both sides keeps the
    /// comparison conservative.
    fn timed<F: FnMut()>(mut f: F) -> f64 {
        for _ in 0..WARMUP {
            f();
        }
        let mut best = f64::INFINITY;
        for _ in 0..ITERS {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    }

    fn host_ramp(n: usize) -> Tensor<f32, 1> {
        Tensor::<f32, 1>::from_shape_fn([n], |[i]| (i % 1000) as f32 - 500.0)
    }

    pub fn run() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no CUDA device: {e}");
                return;
            }
        };
        let stream = ctx.default_stream();
        let props = ctx.name().unwrap_or_else(|_| "unknown".into());
        println!("device: {props}");
        println!("warmup {WARMUP}, {ITERS} timed iters\n");

        println!(
            "{:<18} {:>7} {:>11} {:>12} {:>13} {:>10} {:>12}",
            "op", "size", "CPU ms", "GPU kern ms", "GPU trip ms", "kern x", "trip x"
        );
        println!("{}", "-".repeat(90));

        for &(label, n) in SIZES {
            // ── unary (relu) ──────────────────────────────────────────────
            {
                let src = host_ramp(n);
                let mut dst = Tensor::<f32, 1>::zeros([n]);
                let cpu = timed(|| {
                    ops::apply_unary(&src, &mut dst, UnaryOp::Relu).unwrap();
                });

                match (
                    src.to_cuda(&stream),
                    Tensor::<f32, 1>::zeros([n]).to_cuda(&stream),
                ) {
                    (Ok(d_src), Ok(mut d_dst)) => {
                        let kern = timed(|| {
                            ops::apply_unary(&d_src, &mut d_dst, UnaryOp::Relu).unwrap();
                            stream.synchronize().unwrap();
                        });
                        let trip = timed(|| {
                            let a = src.to_cuda(&stream).unwrap();
                            let mut o = Tensor::<f32, 1>::zeros([n]).to_cuda(&stream).unwrap();
                            ops::apply_unary(&a, &mut o, UnaryOp::Relu).unwrap();
                            let _ = o.to_host(&stream).unwrap();
                        });
                        row("unary relu", label, cpu, kern, trip);
                    }
                    _ => println!(
                        "{:<18} {:>7}   device alloc failed (skipped)",
                        "unary relu", label
                    ),
                }
            }

            // ── binary (add, mul) ─────────────────────────────────────────
            for (name, op) in [("binary add", BinaryOp::Add), ("binary mul", BinaryOp::Mul)] {
                let a = host_ramp(n);
                let b = host_ramp(n);
                let mut dst = Tensor::<f32, 1>::zeros([n]);
                let cpu = timed(|| {
                    ops::apply_binary(&a, &b, &mut dst, op).unwrap();
                });

                match (
                    a.to_cuda(&stream),
                    b.to_cuda(&stream),
                    Tensor::<f32, 1>::zeros([n]).to_cuda(&stream),
                ) {
                    (Ok(d_a), Ok(d_b), Ok(mut d_dst)) => {
                        let kern = timed(|| {
                            ops::apply_binary(&d_a, &d_b, &mut d_dst, op).unwrap();
                            stream.synchronize().unwrap();
                        });
                        let trip = timed(|| {
                            let x = a.to_cuda(&stream).unwrap();
                            let y = b.to_cuda(&stream).unwrap();
                            let mut o = Tensor::<f32, 1>::zeros([n]).to_cuda(&stream).unwrap();
                            ops::apply_binary(&x, &y, &mut o, op).unwrap();
                            let _ = o.to_host(&stream).unwrap();
                        });
                        row(name, label, cpu, kern, trip);
                    }
                    _ => println!("{name:<18} {label:>7}   device alloc failed (skipped)"),
                }
            }

            // ── reduce (sum) ──────────────────────────────────────────────
            {
                let src = host_ramp(n);
                let cpu = timed(|| {
                    let _ = ops::reduce(&src, ReduceOp::Sum).unwrap();
                });
                match src.to_cuda(&stream) {
                    Ok(d_src) => {
                        // `ops::reduce` already synchronizes before reading back.
                        let kern = timed(|| {
                            let _ = ops::reduce(&d_src, ReduceOp::Sum).unwrap();
                        });
                        let trip = timed(|| {
                            let a = src.to_cuda(&stream).unwrap();
                            let _ = ops::reduce(&a, ReduceOp::Sum).unwrap();
                        });
                        row("reduce sum", label, cpu, kern, trip);
                    }
                    Err(_) => println!(
                        "{:<18} {:>7}   device alloc failed (skipped)",
                        "reduce sum", label
                    ),
                }
            }
        }

        determinism_probe(&stream);
    }

    fn row(op: &str, size: &str, cpu: f64, kern: f64, trip: f64) {
        println!(
            "{op:<18} {size:>7} {cpu:>11.3} {kern:>12.3} {trip:>13.3} {:>9.1}x {:>11.1}x",
            cpu / kern,
            cpu / trip
        );
    }

    /// The GPU reduction accumulates block partials with `atomicAdd` on f32,
    /// so the summation order varies between launches. Float addition is not
    /// associative — this reports whether repeated runs over identical input
    /// actually differ, and by how much against the CPU result.
    fn determinism_probe(stream: &std::sync::Arc<cudarc::driver::CudaStream>) {
        const N: usize = 10_000_000;
        // Adversarial input: interleaving values ~10 orders of magnitude apart
        // makes the sum order-sensitive. A ramp would not — its total is exactly
        // representable, so any ordering gives the same f32 and the probe would
        // report determinism it has not actually tested.
        let src =
            Tensor::<f32, 1>::from_shape_fn([N], |[i]| if i % 2 == 0 { 1.0e7 } else { 1.0e-3 });
        let cpu = match ops::reduce(&src, ReduceOp::Sum) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\nreduce determinism probe skipped: {e}");
                return;
            }
        };
        let Ok(dev) = src.to_cuda(stream) else {
            eprintln!("\nreduce determinism probe skipped: device alloc failed");
            return;
        };

        let mut seen: Vec<f32> = Vec::new();
        for _ in 0..20 {
            if let Ok(v) = ops::reduce(&dev, ReduceOp::Sum) {
                if !seen.contains(&v) {
                    seen.push(v);
                }
            }
        }
        println!("\n── reduce sum determinism (n = {N}, 20 runs) ──");
        println!("CPU sum:            {cpu:.6}");
        println!("distinct GPU sums:  {}", seen.len());
        if let (Some(&lo), Some(&hi)) = (
            seen.iter().min_by(|a, b| a.total_cmp(b)),
            seen.iter().max_by(|a, b| a.total_cmp(b)),
        ) {
            println!("GPU range:          {lo:.6} .. {hi:.6}");
            println!(
                "max |GPU - CPU|:    {:.6}",
                (hi - cpu).abs().max((lo - cpu).abs())
            );
        }
    }
}
