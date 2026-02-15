# [Feature]: Support RANSAC Variants for Two-View Pose Estimation

> **Note**: This document describes a feature request for supporting RANSAC variants (LO-RANSAC, GC-RANSAC) in the two-view pose estimation module. This feature was discussed in [PR #688](https://github.com/kornia/kornia-rs/pull/688#issuecomment-3867423085).

---

## 🚀 Feature Description

Add support for advanced RANSAC variants in the `kornia-3d` pose estimation module, specifically:

1. **LO-RANSAC (Locally Optimized RANSAC)**: Performs local optimization on promising hypotheses to improve model quality before final selection.

2. **GC-RANSAC (Graph-Cut RANSAC)**: Uses graph-cut optimization for spatial consistency in inlier selection, improving robustness in structured scenes.

3. **PROSAC (Progressive Sample Consensus)**: Leverages match quality scores for more efficient sampling, reducing iterations needed for convergence.

These variants would enhance the existing `ransac_fundamental` and `ransac_homography` functions in `kornia-3d/src/pose/twoview.rs`, as well as the PnP RANSAC in `kornia-3d/src/pnp/ransac.rs`.

---

## 📂 Feature Category

Geometry

---

## 💡 Motivation

The current RANSAC implementation in `kornia-3d` uses a basic vanilla RANSAC approach. While functional, this has limitations:

1. **Efficiency**: Vanilla RANSAC may require many iterations to find a good model, especially with high outlier ratios.

2. **Model Quality**: Without local optimization, the final model may not be as refined as it could be.

3. **Spatial Consistency**: Vanilla RANSAC treats points independently, missing opportunities to leverage spatial structure.

As discussed by @ducha-aiki in [PR #688](https://github.com/kornia/kornia-rs/pull/688#issuecomment-3867423085):
> "yes - LO-RANSAC, GC-RANSAC. Overall I'd mimic https://github.com/PoseLib/PoseLib"

These variants are standard in production-quality pose estimation pipelines and would significantly improve robustness and efficiency for real-world applications like:
- Visual SLAM initialization
- Structure from Motion (SfM) pipelines
- Image stitching and panorama creation
- Augmented reality applications

---

## 💭 Proposed Solution

### 1. Extend `RansacParams` Configuration

Add configuration options to support different RANSAC strategies:

```rust
/// RANSAC method variants
#[derive(Clone, Copy, Debug, Default)]
pub enum RansacMethod {
    /// Standard RANSAC (current implementation)
    #[default]
    Standard,
    /// Locally Optimized RANSAC - refines promising hypotheses
    LoRansac {
        /// Number of local optimization iterations
        lo_iterations: usize,
        /// Inlier threshold multiplier for LO step
        lo_threshold_mult: f64,
    },
    /// Graph-Cut RANSAC - spatial consistency via graph cuts
    GcRansac {
        /// Neighborhood radius for graph construction
        neighborhood_radius: f64,
        /// Spatial coherence weight
        spatial_weight: f64,
    },
    /// Progressive RANSAC - quality-guided sampling
    Prosac {
        /// Maximum PROSAC-specific iterations before falling back to uniform sampling
        max_prosac_iterations: usize,
    },
}

#[derive(Clone, Debug)]
pub struct RansacParams {
    /// Maximum number of RANSAC iterations
    pub max_iterations: usize,
    /// Minimum number of iterations before early termination
    pub min_iterations: usize,
    /// Inlier threshold (pixel error)
    pub threshold: f64,
    /// Minimum number of inliers required for acceptance
    pub min_inliers: usize,
    /// Target success probability for adaptive iteration count
    pub confidence: f64,
    /// RANSAC method variant
    pub method: RansacMethod,
    /// Optional RNG seed for deterministic runs
    pub random_seed: Option<u64>,
}
```

### 2. Implement LO-RANSAC

LO-RANSAC adds a local optimization step when a new best model is found:

```rust
/// Local optimization step for LO-RANSAC
fn local_optimize<M>(
    model: &M,
    inliers: &[bool],
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &LoRansacParams,
    estimator: impl Fn(&[Vec2F64], &[Vec2F64]) -> Result<M, Error>,
) -> Result<M, Error> {
    // 1. Collect inlier correspondences
    // 2. Re-estimate model using all inliers (or a larger subset)
    // 3. Optionally iterate with expanded inlier set
    // 4. Return refined model
}
```

### 3. Implement PROSAC Sampling

PROSAC assumes correspondences are sorted by quality (e.g., descriptor distance):

```rust
/// PROSAC sampling for quality-guided hypothesis generation
fn prosac_sample(
    n_total: usize,
    sample_size: usize,
    iteration: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    // Progressive sampling that prioritizes high-quality matches early
}
```

### 4. API Consistency

Maintain backward compatibility - existing code using `RansacParams::default()` should continue to work with standard RANSAC behavior.

---

## 📚 Library Reference

This feature is based on established implementations and research:

1. **PoseLib** (C++): https://github.com/PoseLib/PoseLib
   - Reference implementation recommended by @ducha-aiki
   - Provides LO-RANSAC with non-linear refinement

2. **OpenCV USAC Framework**: 
   - Implements multiple RANSAC variants (RANSAC, PROSAC, LO-RANSAC, GC-RANSAC, MAGSAC)
   - Reference: `cv::usac` module

3. **Academic Papers**:
   - LO-RANSAC: Chum, O., Matas, J., & Kittler, J. (2003). "Locally optimized RANSAC"
   - GC-RANSAC: Barath, D., & Matas, J. (2018). "Graph-Cut RANSAC"
   - PROSAC: Chum, O., & Matas, J. (2005). "Matching with PROSAC - Progressive Sample Consensus"
   - MAGSAC: Barath, D., et al. (2019). "MAGSAC: marginalizing sample consensus"

4. **Rust Ecosystem**:
   - Consider integration patterns from `cv-core` crate's robust estimation traits

---

## 🔄 Alternatives Considered

1. **External Dependency**: Using an existing RANSAC library crate
   - Rejected: Would add external dependency and may not integrate well with kornia-algebra types

2. **Only LO-RANSAC**: Implementing just LO-RANSAC as the simplest improvement
   - Considered: Could be a good first step, with other variants added later

3. **Generic RANSAC Trait**: Creating a fully generic RANSAC framework
   - Considered: May add complexity; focused implementations may be more practical initially

---

## 🎯 Use Cases

1. **Visual SLAM Initialization**: Two-view initialization with robust pose recovery from potentially noisy feature matches

2. **Structure from Motion**: Essential/fundamental matrix estimation in SfM pipelines with varying outlier ratios

3. **Image Stitching**: Homography estimation for panorama creation with robust outlier handling

4. **Augmented Reality**: Real-time pose estimation requiring efficient and robust model fitting

5. **Robotics**: Visual odometry and localization with reliable relative pose estimation

---

## 📝 Additional Context

### Current Implementation Status

The current RANSAC implementation in `twoview.rs` provides:
- Basic RANSAC loop with configurable iterations
- Support for fundamental matrix (8-point) and homography (4-point) estimation
- Sampson distance for fundamental matrix scoring
- Reprojection error for homography scoring

### Suggested Implementation Phases

**Phase 1**: LO-RANSAC
- Add local optimization after finding new best model
- Non-linear refinement using inliers
- Backward compatible with existing API

**Phase 2**: PROSAC
- Add quality-guided sampling option
- Requires correspondences sorted by match quality
- Falls back to uniform sampling after max_prosac_iterations

**Phase 3**: GC-RANSAC (optional)
- More complex implementation requiring spatial neighborhood graph
- May require additional data structures

### Related Issues/PRs

- PR #688: Two-view initialization implementation (merged)
- Discussion: https://github.com/kornia/kornia-rs/pull/688#issuecomment-3867423085

---

## 🤝 Contribution Intent

- [ ] I plan to submit a PR to implement this feature
- [x] I'm requesting this feature but not planning to implement it

---

## Implementation Notes

When implementing, consider:

1. **Trait-based design** for swappable RANSAC strategies
2. **Benchmark suite** to compare variant performance
3. **Integration with existing `kornia-algebra::optim`** for non-linear refinement in LO step
4. **Python bindings** exposure in `kornia-py` for the new parameters
