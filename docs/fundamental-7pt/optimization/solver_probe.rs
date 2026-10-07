use kornia_3d::{
    pose::fundamental_7point,
    ransac::{estimators::FundamentalEstimator, Estimator, Match2d2d},
};
use kornia_algebra::Vec2F64;
use std::io::{self, BufRead, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rounds = std::env::args()
        .nth(1)
        .and_then(|n| n.parse::<usize>().ok());
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let pairs: Vec<[f64; 4]> = serde_json::from_str(&line?)?;
        let x1: Vec<_> = pairs.iter().map(|p| Vec2F64::new(p[0], p[1])).collect();
        let x2: Vec<_> = pairs.iter().map(|p| Vec2F64::new(p[2], p[3])).collect();
        if let Some(repetitions) = rounds {
            let samples: Vec<_> = x1
                .iter()
                .zip(&x2)
                .map(|(&a, &b)| Match2d2d::new(a, b))
                .collect();
            let mut models = Vec::with_capacity(3);
            let t = std::time::Instant::now();
            for _ in 0..repetitions {
                models.clear();
                FundamentalEstimator.fit(std::hint::black_box(&samples), &mut models);
                std::hint::black_box(&models);
            }
            let fit_ns = t.elapsed().as_nanos() as f64 / repetitions as f64;
            let count = models.len();
            let t = std::time::Instant::now();
            for _ in 0..repetitions {
                std::hint::black_box(fundamental_7point(
                    std::hint::black_box(&x1),
                    std::hint::black_box(&x2),
                )?);
            }
            let public_ns = t.elapsed().as_nanos() as f64 / repetitions as f64;
            serde_json::to_writer(
                &mut output,
                &serde_json::json!({"fit_ns":fit_ns,"public_ns":public_ns,"count":count}),
            )?;
        } else {
            let models: Vec<[f64; 9]> = fundamental_7point(&x1, &x2)
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect();
            serde_json::to_writer(&mut output, &models)?;
        }
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}
