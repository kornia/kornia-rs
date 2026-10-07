use kornia_3d::ransac::{
    estimators::{Fundamental8PointEstimator, FundamentalEstimator},
    run, Consensus, ConsensusOutcome, Estimator, Match2d2d, RansacConfig, UniformSampler,
};
use kornia_algebra::{Mat3F64, Vec2F64};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
struct Policy {
    msac: bool,
    fixed: bool,
}
impl Consensus for Policy {
    fn consensus(&self, residuals: &[f64], mask: &mut Vec<bool>) -> ConsensusOutcome {
        mask.clear();
        let mut count = 0;
        let mut score = 0.;
        for &r in residuals {
            let inside = r < 0.25;
            mask.push(inside);
            count += inside as usize;
            if inside {
                score += if self.msac { 1. - r / 0.25 } else { 1. };
            }
        }
        ConsensusOutcome {
            score,
            inlier_count: if self.fixed { 0 } else { count },
        }
    }
    fn threshold(&self) -> Option<f64> {
        if !self.msac && !self.fixed {
            Some(0.25)
        } else {
            None
        }
    }
}
fn execute<E: Estimator<Model = Mat3F64, Sample = Match2d2d>>(
    e: E,
    samples: &[Match2d2d],
    seed: u64,
    budget: u32,
    confidence: f64,
    mode: &str,
) -> Value {
    let policy = Policy {
        msac: mode.contains("msac"),
        fixed: mode.contains("fixed"),
    };
    let cfg = RansacConfig {
        max_iters: budget,
        confidence,
        lo_every: if mode.contains("lo") { 1 } else { 0 },
        ..Default::default()
    };
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed));
    let start = std::time::Instant::now();
    let r = run(&e, &policy, &mut sampler, samples, &cfg);
    let ns = start.elapsed().as_nanos();
    json!({"solver":E::SAMPLE_SIZE,"mode":mode,"model":r.model.map(|m|m.to_cols_array()),"mask":r.inliers,"draws":r.num_iters,"score":r.score,"ns":ns})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let input: Value = serde_json::from_str(&line?)?;
        let pairs: Vec<[f64; 4]> = serde_json::from_value(input["matches"].clone())?;
        let samples: Vec<_> = pairs
            .iter()
            .map(|p| Match2d2d::new(Vec2F64::new(p[0], p[1]), Vec2F64::new(p[2], p[3])))
            .collect();
        let seed = input["seed"].as_u64().ok_or("seed")?;
        let budget = input["budget"].as_u64().ok_or("budget")? as u32;
        let confidence = input["confidence"].as_f64().ok_or("confidence")?;
        let mut rows = Vec::new();
        let modes: Vec<&str> = input["modes"]
            .as_array()
            .map(|v| v.iter().filter_map(|s| s.as_str()).collect())
            .unwrap_or_else(|| {
                vec![
                    "count_adaptive",
                    "count_fixed",
                    "msac_adaptive",
                    "msac_fixed",
                    "count_lo",
                    "msac_lo",
                ]
            });
        for round in 0..input["rounds"].as_u64().unwrap_or(1) {
            for &mode in &modes {
                if (seed + round) % 2 == 0 {
                    rows.push(execute(
                        FundamentalEstimator,
                        &samples,
                        seed,
                        budget,
                        confidence,
                        mode,
                    ));
                    rows.push(execute(
                        Fundamental8PointEstimator,
                        &samples,
                        seed,
                        budget,
                        confidence,
                        mode,
                    ));
                } else {
                    rows.push(execute(
                        Fundamental8PointEstimator,
                        &samples,
                        seed,
                        budget,
                        confidence,
                        mode,
                    ));
                    rows.push(execute(
                        FundamentalEstimator,
                        &samples,
                        seed,
                        budget,
                        confidence,
                        mode,
                    ));
                }
            }
        }
        serde_json::to_writer(
            &mut writer,
            &json!({"pair":input["pair"],"seed":seed,"budget":budget,"confidence":confidence,"rows":rows}),
        )?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}
