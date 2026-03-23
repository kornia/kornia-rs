use kornia_algebra::{SE3F32, Vec3AF32};

/// Calculate Absolute Trajectory Error (ATE) between estimated and ground truth poses.
/// Assumes both lists are aligned in time.
pub fn ate(estimated: &[SE3F32], ground_truth: &[SE3F32]) -> f32 {
    if estimated.is_empty() || estimated.len() != ground_truth.len() {
        return 0.0;
    }

    let mut sum_sq_err = 0.0;
    for (est, gt) in estimated.iter().zip(ground_truth.iter()) {
        let diff = gt.inverse() * *est;
        let t_err = diff.t.length();
        sum_sq_err += t_err * t_err;
    }

    (sum_sq_err / estimated.len() as f32).sqrt()
}

/// Calculate Relative Pose Error (RPE) for a given time interval.
pub fn rpe(estimated: &[SE3F32], ground_truth: &[SE3F32], delta: usize) -> f32 {
    if estimated.len() <= delta || estimated.len() != ground_truth.len() {
        return 0.0;
    }

    let mut sum_sq_err = 0.0;
    let count = estimated.len() - delta;

    for i in 0..count {
        let est_rel = estimated[i].inverse() * estimated[i + delta];
        let gt_rel = ground_truth[i].inverse() * ground_truth[i + delta];
        let diff = gt_rel.inverse() * est_rel;
        let t_err = diff.t.length();
        sum_sq_err += t_err * t_err;
    }

    (sum_sq_err / count as f32).sqrt()
}
