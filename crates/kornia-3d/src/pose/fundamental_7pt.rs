//! Minimal seven-point fundamental-matrix solver.

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

use crate::pose::fundamental::FundamentalError;

/// Smallest Gauss-Jordan pivot accepted, relative to the unit-scale rows of
/// the Hartley-normalized design matrix.
const PIVOT_EPS: f64 = 1e-12;
const ROOT_EPS: f64 = 32.0 * f64::EPSILON;

/// Estimate every real fundamental matrix consistent with seven correspondences.
///
/// The points are Hartley normalized before the two-dimensional null space of
/// the seven-point design matrix is found.  The rank-two constraint is then
/// imposed exactly by solving its cubic determinant equation.
///
/// # Arguments
///
/// * `x1` - Exactly seven finite points in the first image.
/// * `x2` - Their corresponding finite points in the second image.
///
/// # Returns
///
/// All distinct finite, Frobenius-normalized real fundamental matrices.  A
/// seven-point sample may have one, two, or three such matrices.
///
/// # Errors
///
/// Returns [`FundamentalError::InvalidInput`] when the inputs are not exactly
/// seven finite correspondences, and [`FundamentalError::DegenerateConfiguration`]
/// when they do not determine a finite fundamental matrix.
///
/// # Example
///
/// ```no_run
/// use kornia_3d::pose::fundamental_7point;
/// use kornia_algebra::Vec2F64;
///
/// let x1 = [
///     Vec2F64::new(10.0, 20.0), Vec2F64::new(30.0, 40.0),
///     Vec2F64::new(50.0, 20.0), Vec2F64::new(60.0, 80.0),
///     Vec2F64::new(15.0, 75.0), Vec2F64::new(90.0, 30.0),
///     Vec2F64::new(35.0, 60.0),
/// ];
/// let x2 = [
///     Vec2F64::new(11.0, 18.0), Vec2F64::new(33.0, 38.0),
///     Vec2F64::new(54.0, 19.0), Vec2F64::new(65.0, 78.0),
///     Vec2F64::new(17.0, 70.0), Vec2F64::new(97.0, 28.0),
///     Vec2F64::new(39.0, 58.0),
/// ];
/// let models = fundamental_7point(&x1, &x2)?;
/// # Ok::<(), kornia_3d::pose::FundamentalError>(())
/// ```
pub fn fundamental_7point(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
) -> Result<Vec<Mat3F64>, FundamentalError> {
    let mut models = [Mat3F64::ZERO; 3];
    let count = fundamental_7point_into(x1, x2, &mut models)?;
    Ok(models[..count].to_vec())
}

/// Estimate fundamental matrices into a caller-provided fixed-capacity buffer.
///
/// The seven-point problem has at most three finite real solutions. `models`
/// therefore always has room for every result and this entry point performs no
/// heap allocation.
///
/// # Arguments
///
/// * `x1` - Exactly seven finite points in the first image.
/// * `x2` - Corresponding finite points in the second image.
/// * `models` - Buffer whose result prefix is overwritten on success.
///
/// # Returns
///
/// The number of distinct matrices written to the buffer, from one to three.
///
/// # Errors
///
/// Returns the same invalid-input and degeneracy errors as [`fundamental_7point`].
pub(crate) fn fundamental_7point_into(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    models: &mut [Mat3F64; 3],
) -> Result<usize, FundamentalError> {
    let (Ok(x1), Ok(x2)) = (<&[Vec2F64; 7]>::try_from(x1), <&[Vec2F64; 7]>::try_from(x2)) else {
        return Err(FundamentalError::InvalidInput);
    };
    if x1
        .iter()
        .chain(x2)
        .any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return Err(FundamentalError::InvalidInput);
    }

    #[cfg(target_arch = "x86_64")]
    if kornia_imgproc::simd::cpu_features().has_avx2 && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: AVX2 and FMA support is checked at runtime; the solver only
        // uses fixed-size arrays and has no other preconditions.
        return unsafe { solve_avx2_fma(x1, x2, models) };
    }

    // aarch64 always has fused multiply-add; elsewhere stay with plain arithmetic.
    solve::<{ cfg!(target_arch = "aarch64") }>(x1, x2, models)
}

/// Compiles the whole solver with AVX2/FMA enabled, so `mul_add` lowers to a
/// single instruction and the fixed-size loops vectorize.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn solve_avx2_fma(
    x1: &[Vec2F64; 7],
    x2: &[Vec2F64; 7],
    models: &mut [Mat3F64; 3],
) -> Result<usize, FundamentalError> {
    solve::<true>(x1, x2, models)
}

/// `a * b + c`, fused when the instantiation targets hardware FMA.
#[inline(always)]
fn fma<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Nine-term dot product with a balanced reduction tree. The seven-point
/// solver is latency bound, so short dependency chains matter more than
/// operation count.
#[inline(always)]
fn dot9<const FMA: bool>(a: &[f64; 9], b: &[f64; 9]) -> f64 {
    let s0 = fma::<FMA>(a[1], b[1], a[0] * b[0]);
    let s1 = fma::<FMA>(a[3], b[3], a[2] * b[2]);
    let s2 = fma::<FMA>(a[5], b[5], a[4] * b[4]);
    let s3 = fma::<FMA>(a[7], b[7], a[6] * b[6]);
    fma::<FMA>(a[8], b[8], (s0 + s1) + (s2 + s3))
}

#[inline(always)]
fn sum7(v: &[f64; 7]) -> f64 {
    ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + v[6])
}

#[inline(always)]
fn solve<const FMA: bool>(
    x1: &[Vec2F64; 7],
    x2: &[Vec2F64; 7],
    models: &mut [Mat3F64; 3],
) -> Result<usize, FundamentalError> {
    let h1 = Hartley::new(x1).ok_or(FundamentalError::DegenerateConfiguration)?;
    let h2 = Hartley::new(x2).ok_or(FundamentalError::DegenerateConfiguration)?;
    let [f0, f1] =
        null_space_7x9::<FMA>(&h1, &h2).ok_or(FundamentalError::DegenerateConfiguration)?;

    let coefficients = determinant_polynomial(&f0, &f1);
    let coefficient_scale = coefficients
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if !coefficient_scale.is_finite() || coefficient_scale <= ROOT_EPS {
        return Err(FundamentalError::DegenerateConfiguration);
    }
    let (roots, root_count) = conditioned_roots(coefficients);

    // Candidates are independent; build them all before the order-dependent
    // deduplication so their latency chains overlap.
    let mut candidates = [None; 3];
    for (candidate, &(_, alpha, beta)) in candidates.iter_mut().zip(&roots[..root_count]) {
        let f = std::array::from_fn(|i| fma::<FMA>(alpha, f0[i], beta * f1[i]));
        *candidate = validated_model::<FMA>(&f, &h1, &h2);
    }
    let mut model_count = 0;
    for model in candidates.iter().flatten() {
        if !models[..model_count]
            .iter()
            .any(|other| matrices_proportional(other, model))
        {
            models[model_count] = *model;
            model_count += 1;
        }
    }

    if model_count == 0 {
        Err(FundamentalError::DegenerateConfiguration)
    } else {
        Ok(model_count)
    }
}

/// Choose a projective parameterization with a well-conditioned leading term.
/// The four determinants are already available from the homogeneous cubic.
/// Preserve the original pencil's candidate order so count ties in RANSAC
/// remain independent of this numerical conditioning choice.
#[inline(always)]
fn conditioned_roots([a, b, c, d]: [f64; 4]) -> ([(f64, f64, f64); 3], usize) {
    let choices = [
        [a, b, c, d],
        [d, c, b, a],
        [a + b + c + d, b + 2.0 * c + 3.0 * d, c + 3.0 * d, d],
        [a - b + c - d, b - 2.0 * c + 3.0 * d, c - 3.0 * d, d],
    ];
    let mut choice = 0;
    for i in 1..4 {
        if choices[i][0].abs() > choices[choice][0].abs() {
            choice = i;
        }
    }
    // A nonzero homogeneous cubic cannot vanish at all four distinct
    // projective directions. The selected polynomial therefore has full
    // degree, including when the original basis had a root at infinity.
    let (roots, count) = real_polynomial_roots(choices[choice]);
    let mut keyed = [(f64::INFINITY, 0.0, 0.0); 3];
    for (entry, root) in keyed.iter_mut().zip(roots).take(count) {
        let (alpha, beta) = match choice {
            0 => (1.0, root),
            1 => (root, 1.0),
            2 => (1.0 + root, root),
            _ => (1.0 - root, root),
        };
        let key = if alpha == 0.0 {
            f64::INFINITY
        } else {
            beta / alpha
        };
        *entry = (key, alpha, beta);
    }
    // Three-element sorting network; unused entries stay last.
    let swap_if_less = |v: &mut [(f64, f64, f64); 3], i: usize, j: usize| {
        if v[j].0.total_cmp(&v[i].0).is_lt() {
            v.swap(i, j);
        }
    };
    if count > 1 {
        swap_if_less(&mut keyed, 0, 1);
        if count > 2 {
            swap_if_less(&mut keyed, 1, 2);
            swap_if_less(&mut keyed, 0, 1);
        }
    }
    (keyed, count)
}

/// Hartley normalization of seven points: centroid at the origin and mean
/// distance √2, represented by `T = [[s, 0, -s c_x], [0, s, -s c_y], [0, 0, 1]]`.
struct Hartley {
    x: [f64; 7],
    y: [f64; 7],
    scale: f64,
    cx: f64,
    cy: f64,
}

impl Hartley {
    #[inline(always)]
    fn new(points: &[Vec2F64; 7]) -> Option<Self> {
        let xs: [f64; 7] = std::array::from_fn(|i| points[i].x);
        let ys: [f64; 7] = std::array::from_fn(|i| points[i].y);
        let cx = sum7(&xs) / 7.0;
        let cy = sum7(&ys) / 7.0;
        let mut x = [0.0; 7];
        let mut y = [0.0; 7];
        let mut distance = [0.0; 7];
        for i in 0..7 {
            x[i] = xs[i] - cx;
            y[i] = ys[i] - cy;
            distance[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
        }
        let mean_distance = sum7(&distance) / 7.0;
        if !mean_distance.is_finite() || mean_distance <= f64::EPSILON {
            return None;
        }
        let scale = std::f64::consts::SQRT_2 / mean_distance;
        for i in 0..7 {
            x[i] *= scale;
            y[i] *= scale;
        }
        Some(Self {
            x,
            y,
            scale,
            cx,
            cy,
        })
    }
}

/// Compute an orthonormal basis of the two-dimensional null space of A.
///
/// Gauss-Jordan elimination of the 7x9 design matrix with column pivoting:
/// each row pivots on its largest coefficient among the columns not yet
/// used, so the two free coefficients are chosen adaptively rather than
/// fixed in advance. Pivot choices use branch-free packed keys, and rows are
/// combined without division (`p·row_j - row_j[c]·row_k`, rescaled by the
/// exact power of two of `p`), which keeps the per-step dependency chain
/// short while staying within a factor 2^7 of ordinary elimination. The two
/// resulting null vectors are orthonormalized so the determinant pencil is
/// well scaled.
#[inline(always)]
fn null_space_7x9<const FMA: bool>(h1: &Hartley, h2: &Hartley) -> Option<[[f64; 9]; 2]> {
    // Rows are padded to three full four-lane vectors: every update is then
    // stored and reloaded with the same widths, which avoids store-forwarding
    // stalls between elimination steps.
    let mut a = [[0.0f64; 12]; 7];
    for (row, i) in a.iter_mut().zip(0..7) {
        let (x, y, xp, yp) = (h1.x[i], h1.y[i], h2.x[i], h2.y[i]);
        *row = [
            xp * x,
            xp * y,
            xp,
            yp * x,
            yp * y,
            yp,
            x,
            y,
            1.0,
            0.0,
            0.0,
            0.0,
        ];
    }

    let mut used = 0u64;
    let mut pivot_columns = [0usize; 7];
    // Accumulated power-of-two-normalized scale of the rows still to pivot.
    let mut growth = 1.0f64;
    for k in 0..7 {
        let row = a[k];
        // Non-negative doubles order like their bit patterns. Packing the
        // column index into the low mantissa bits turns the pivot search
        // into an integer maximum; used columns are masked to zero.
        let key = |column: usize| -> u64 {
            let unused = ((used >> column) & 1).wrapping_sub(1);
            ((row[column].abs().to_bits() & !15) | column as u64) & unused
        };
        let best = key(0)
            .max(key(1))
            .max(key(2).max(key(3)))
            .max(key(4).max(key(5)).max(key(6).max(key(7))))
            .max(key(8));
        let best_magnitude = f64::from_bits(best & !15);
        if best_magnitude.is_nan() || best_magnitude <= PIVOT_EPS * growth {
            return None;
        }
        let column = (best & 15) as usize;
        let pivot = row[column];
        used |= 1 << column;
        pivot_columns[k] = column;

        // 2^-exponent(pivot) is exact and maps the pivot into ±[1, 2).
        let pow2 = f64::from_bits((2046 - ((pivot.to_bits() >> 52) & 0x7ff)) << 52);
        let scaled_pivot = pivot * pow2;
        growth *= scaled_pivot.abs();
        let mut pivot_row = row;
        for value in &mut pivot_row {
            *value *= pow2;
        }
        for (j, other) in a.iter_mut().enumerate() {
            if j != k {
                let factor = other[column];
                for (value, &p) in other.iter_mut().zip(&pivot_row) {
                    *value = fma::<FMA>(scaled_pivot, *value, -(factor * p));
                }
            }
        }
    }

    let free = !used & 0x1ff;
    let free0 = free.trailing_zeros() as usize;
    let free1 = (free & (free - 1)).trailing_zeros() as usize;
    // Null vectors in pivot order: entry k belongs to coefficient
    // `pivot_columns[k]`; entries 7 and 8 are the two free coefficients.
    let mut x = [0.0f64; 9];
    let mut y = [0.0f64; 9];
    for k in 0..7 {
        let inv_pivot = 1.0 / a[k][pivot_columns[k]];
        x[k] = -a[k][free0] * inv_pivot;
        y[k] = -a[k][free1] * inv_pivot;
    }
    x[7] = 1.0;
    y[8] = 1.0;

    // Gram-Schmidt on the pair; dot products do not depend on the order.
    let xx = dot9::<FMA>(&x, &x);
    let ratio = dot9::<FMA>(&x, &y) / xx;
    let w: [f64; 9] = std::array::from_fn(|i| fma::<FMA>(-ratio, x[i], y[i]));
    let inv_x = 1.0 / xx.sqrt();
    let inv_w = 1.0 / dot9::<FMA>(&w, &w).sqrt();
    if !inv_x.is_finite() || !inv_w.is_finite() {
        return None;
    }
    let mut basis = [[0.0f64; 9]; 2];
    for k in 0..7 {
        basis[0][pivot_columns[k]] = x[k] * inv_x;
        basis[1][pivot_columns[k]] = w[k] * inv_w;
    }
    basis[0][free0] = inv_x;
    basis[1][free0] = w[7] * inv_w;
    basis[1][free1] = w[8] * inv_w;
    Some(basis)
}

/// Validate one root of the pencil and map it to a pixel-space matrix.
#[inline(always)]
fn validated_model<const FMA: bool>(f: &[f64; 9], t1: &Hartley, t2: &Hartley) -> Option<Mat3F64> {
    let fnorm = vector_norm::<FMA>(f);
    if !fnorm.is_finite()
        || fnorm <= f64::MIN_POSITIVE
        || determinant(f).abs() > 1e-8 * fnorm.powi(3)
        || !has_rank_at_least_two(f, fnorm)
    {
        return None;
    }
    frobenius_normalize::<FMA>(denormalize(f, t1, t2))
}

/// Check that a rank-two candidate has a numerically nonzero two-by-two minor.
///
/// The determinant constraint alone also admits rank-one matrices, which do not
/// represent a valid fundamental matrix and must not reach RANSAC scoring.
#[inline(always)]
fn has_rank_at_least_two(f: &[f64; 9], norm: f64) -> bool {
    let threshold = 1e-10 * norm * norm;
    for row0 in 0..3 {
        for row1 in (row0 + 1)..3 {
            for col0 in 0..3 {
                for col1 in (col0 + 1)..3 {
                    let minor = f[row0 * 3 + col0] * f[row1 * 3 + col1]
                        - f[row0 * 3 + col1] * f[row1 * 3 + col0];
                    if minor.abs() > threshold {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Coefficients `[a, b, c, d]` of det(f0 + lambda f1).
#[inline(always)]
fn determinant_polynomial(f0: &[f64; 9], f1: &[f64; 9]) -> [f64; 4] {
    let c0 = |f: &[f64; 9]| [f[0], f[3], f[6]];
    let c1 = |f: &[f64; 9]| [f[1], f[4], f[7]];
    let c2 = |f: &[f64; 9]| [f[2], f[5], f[8]];
    let (a0, a1, a2) = (c0(f0), c1(f0), c2(f0));
    let (b0, b1, b2) = (c0(f1), c1(f1), c2(f1));
    [
        determinant_columns(b0, b1, b2),
        determinant_columns(b0, b1, a2)
            + determinant_columns(b0, a1, b2)
            + determinant_columns(a0, b1, b2),
        determinant_columns(b0, a1, a2)
            + determinant_columns(a0, b1, a2)
            + determinant_columns(a0, a1, b2),
        determinant_columns(a0, a1, a2),
    ]
}

#[inline(always)]
fn determinant_columns(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]) -> f64 {
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
}

#[inline(always)]
fn determinant(f: &[f64; 9]) -> f64 {
    f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6])
}

/// Return all distinct finite real roots of a polynomial of degree at most three.
#[inline(always)]
fn real_polynomial_roots([a, b, c, d]: [f64; 4]) -> ([f64; 3], usize) {
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if !scale.is_finite() || scale == 0.0 {
        return ([0.0; 3], 0);
    }
    let [a, b, c, d] = [a / scale, b / scale, c / scale, d / scale];
    let epsilon = ROOT_EPS;
    let (mut roots, root_count) = if a.abs() <= epsilon {
        quadratic_roots(b, c, d, epsilon)
    } else {
        cubic_roots(a, b, c, d)
    };
    // Up to three Newton steps, applied to all roots together so their
    // latency chains overlap. A root whose step is rejected keeps its value
    // and is rejected again, exactly as if its own iteration had stopped.
    for _ in 0..3 {
        let mut moved = false;
        for root in &mut roots[..root_count] {
            let r = *root;
            let value = ((a * r + b) * r + c) * r + d;
            // Once the Horner residual is at its rounding-error floor,
            // Newton can only amplify noise, particularly at a double root.
            let magnitude = ((a.abs() * r.abs() + b.abs()) * r.abs() + c.abs()) * r.abs() + d.abs();
            let derivative = (3.0 * a * r + 2.0 * b) * r + c;
            let next = r - value / derivative;
            let step = value.abs() > 4.0 * f64::EPSILON * magnitude
                && value.is_finite()
                && derivative.is_finite()
                && derivative.abs() > ROOT_EPS
                && next.is_finite();
            *root = if step { next } else { r };
            moved |= step;
        }
        if !moved {
            break;
        }
    }

    // Finite roots in ascending order, without near-duplicates.
    let mut sorted = [f64::INFINITY; 3];
    let mut finite = 0;
    for &root in &roots[..root_count] {
        if root.is_finite() {
            sorted[finite] = root;
            finite += 1;
        }
    }
    let (low, high) = (sorted[0].min(sorted[1]), sorted[0].max(sorted[1]));
    let (middle, top) = (high.min(sorted[2]), high.max(sorted[2]));
    let sorted = [low.min(middle), low.max(middle), top];
    let mut distinct = [0.0; 3];
    let mut write = 0;
    for &root in &sorted[..finite] {
        if write == 0
            || (root - distinct[write - 1]).abs()
                > ROOT_EPS * root.abs().max(distinct[write - 1].abs()).max(1.0)
        {
            distinct[write] = root;
            write += 1;
        }
    }
    (distinct, write)
}

#[inline(always)]
fn quadratic_roots(a: f64, b: f64, c: f64, epsilon: f64) -> ([f64; 3], usize) {
    if a.abs() <= epsilon {
        return if b.abs() <= epsilon {
            ([0.0; 3], 0)
        } else {
            ([-c / b, 0.0, 0.0], 1)
        };
    }
    let discriminant = b * b - 4.0 * a * c;
    let discriminant_tolerance = epsilon * (b * b + (4.0 * a * c).abs());
    if discriminant < -discriminant_tolerance {
        return ([0.0; 3], 0);
    }
    if discriminant.abs() <= discriminant_tolerance {
        return ([-b / (2.0 * a), 0.0, 0.0], 1);
    }
    let sqrt_discriminant = discriminant.sqrt();
    let q = -0.5 * (b + b.signum() * sqrt_discriminant);
    if q == 0.0 {
        (
            [
                (-b + sqrt_discriminant) / (2.0 * a),
                (-b - sqrt_discriminant) / (2.0 * a),
                0.0,
            ],
            2,
        )
    } else {
        ([q / a, c / q, 0.0], 2)
    }
}

#[inline(always)]
fn cubic_roots(a: f64, b: f64, c: f64, d: f64) -> ([f64; 3], usize) {
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    let offset = -b / (3.0 * a);
    let half_q = q * 0.5;
    let third_p = p / 3.0;
    let discriminant = half_q * half_q + third_p * third_p * third_p;
    // Include rounding from the null-space basis, mixed determinant
    // coefficients and projective reparameterization, as well as Cardano
    // cancellation.
    let tolerance = 128.0 * f64::EPSILON * (half_q * half_q + third_p.abs() * third_p * third_p);
    if discriminant > tolerance {
        // Choose the larger-magnitude Cardano term to avoid cancellation,
        // then use u*v = -p/3 instead of taking a second cube root.
        let u = (-half_q - discriminant.sqrt().copysign(half_q)).cbrt();
        let v = if u != 0.0 { -p / (3.0 * u) } else { 0.0 };
        return ([offset + u + v, 0.0, 0.0], 1);
    }
    if discriminant >= -tolerance {
        let u = (-half_q).cbrt();
        return ([offset + 2.0 * u, offset - u, 0.0], 2);
    }
    let radius = 2.0 * (-third_p).sqrt();
    let angle = ((3.0 * q / (2.0 * p)) * (-3.0 / p).sqrt())
        .clamp(-1.0, 1.0)
        .acos()
        / 3.0;
    let (sin, cos) = angle.sin_cos();
    let midpoint = -0.5 * cos;
    let separation = (3.0_f64.sqrt() / 2.0) * sin;
    (
        [
            offset + radius * cos,
            offset + radius * (midpoint + separation),
            offset + radius * (midpoint - separation),
        ],
        3,
    )
}

/// Expand T2^T F T1 using the scale/translation structure of Hartley transforms.
#[inline(always)]
fn denormalize(f: &[f64; 9], t1: &Hartley, t2: &Hartley) -> [f64; 9] {
    let (s1, tx1, ty1) = (t1.scale, -t1.scale * t1.cx, -t1.scale * t1.cy);
    let (s2, tx2, ty2) = (t2.scale, -t2.scale * t2.cx, -t2.scale * t2.cy);
    let row0 = [s2 * f[0], s2 * f[1], s2 * f[2]];
    let row1 = [s2 * f[3], s2 * f[4], s2 * f[5]];
    let row2 = [
        tx2 * f[0] + ty2 * f[3] + f[6],
        tx2 * f[1] + ty2 * f[4] + f[7],
        tx2 * f[2] + ty2 * f[5] + f[8],
    ];
    [
        row0[0] * s1,
        row0[1] * s1,
        row0[0] * tx1 + row0[1] * ty1 + row0[2],
        row1[0] * s1,
        row1[1] * s1,
        row1[0] * tx1 + row1[1] * ty1 + row1[2],
        row2[0] * s1,
        row2[1] * s1,
        row2[0] * tx1 + row2[1] * ty1 + row2[2],
    ]
}

#[inline(always)]
fn vector_norm<const FMA: bool>(values: &[f64; 9]) -> f64 {
    let squared = dot9::<FMA>(values, values);
    if squared.is_finite() && squared >= f64::MIN_POSITIVE {
        return squared.sqrt();
    }
    scaled_norm(values)
}

/// Overflow/underflow-safe fallback for extreme projective roots and coordinates.
#[cold]
fn scaled_norm(values: &[f64; 9]) -> f64 {
    let maximum = values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if maximum == 0.0 {
        return 0.0;
    }
    maximum
        * values
            .iter()
            .map(|value| (value / maximum).powi(2))
            .sum::<f64>()
            .sqrt()
}

/// Row-major entries scaled to unit Frobenius norm, as a matrix.
#[inline(always)]
fn frobenius_normalize<const FMA: bool>(f: [f64; 9]) -> Option<Mat3F64> {
    let norm = vector_norm::<FMA>(&f);
    if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
        return None;
    }
    let inv_norm = 1.0 / norm;
    Some(Mat3F64::from_cols(
        Vec3F64::new(f[0] * inv_norm, f[3] * inv_norm, f[6] * inv_norm),
        Vec3F64::new(f[1] * inv_norm, f[4] * inv_norm, f[7] * inv_norm),
        Vec3F64::new(f[2] * inv_norm, f[5] * inv_norm, f[8] * inv_norm),
    ))
}

#[inline(always)]
fn matrices_proportional(left: &Mat3F64, right: &Mat3F64) -> bool {
    let left: [f64; 9] = (*left).into();
    let right: [f64; 9] = (*right).into();
    let mut same = 0.0;
    let mut opposite = 0.0;
    for (a, b) in left.iter().zip(right) {
        same += (a - b) * (a - b);
        opposite += (a + b) * (a + b);
    }
    same.min(opposite) <= ROOT_EPS * ROOT_EPS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ([Vec2F64; 8], [Vec2F64; 8]) {
        let angle = 0.23_f64;
        let (sin, cos) = angle.sin_cos();
        let points = [
            (-0.7, -0.3, 3.0),
            (0.4, -0.6, 4.2),
            (0.8, 0.5, 2.7),
            (-0.2, 0.9, 3.6),
            (0.1, -0.8, 5.1),
            (-0.9, 0.4, 4.5),
            (0.6, 0.2, 3.3),
            (-0.4, -0.7, 2.9),
        ];
        let mut x1 = [Vec2F64::ZERO; 8];
        let mut x2 = [Vec2F64::ZERO; 8];
        for (i, (x, y, z)) in points.into_iter().enumerate() {
            x1[i] = Vec2F64::new(x / z, y / z);
            let xr = cos * x + sin * z + 0.35;
            let yr = y - 0.11;
            let zr = -sin * x + cos * z + 0.18;
            x2[i] = Vec2F64::new(xr / zr, yr / zr);
        }
        (x1, x2)
    }

    fn epipolar_error(f: &Mat3F64, x1: Vec2F64, x2: Vec2F64) -> f64 {
        Vec3F64::new(x2.x, x2.y, 1.0).dot(*f * Vec3F64::new(x1.x, x1.y, 1.0))
    }

    fn pseudo_random(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn random_sample(seed: &mut u64, trial: usize) -> ([Vec2F64; 8], [Vec2F64; 8]) {
        let angle = 0.08 + 0.017 * trial as f64;
        let (sin, cos) = angle.sin_cos();
        let mut x1 = [Vec2F64::ZERO; 8];
        let mut x2 = [Vec2F64::ZERO; 8];
        for i in 0..8 {
            let x = 2.0 * pseudo_random(seed) - 1.0;
            let y = 2.0 * pseudo_random(seed) - 1.0;
            let z = 3.0 + 3.0 * pseudo_random(seed);
            x1[i] = Vec2F64::new(x / z, y / z);
            let xr = cos * x + sin * z + 0.18 + 0.01 * trial as f64;
            let yr = y - 0.12 + 0.005 * trial as f64;
            let zr = -sin * x + cos * z + 0.23;
            x2[i] = Vec2F64::new(xr / zr, yr / zr);
        }
        (x1, x2)
    }

    #[test]
    fn seven_point_models_fit_sample_and_holdout() {
        let (x1, x2) = sample();
        let models = fundamental_7point(&x1[..7], &x2[..7]).unwrap();
        assert!((1..=3).contains(&models.len()));
        for model in &models {
            assert!(model.determinant().abs() < 1e-9);
            for i in 0..7 {
                assert!(epipolar_error(model, x1[i], x2[i]).abs() < 1e-9);
            }
        }
        assert!(models
            .iter()
            .any(|model| epipolar_error(model, x1[7], x2[7]).abs() < 1e-9));
    }

    #[test]
    fn seven_point_is_stable_under_order_and_coordinate_scale() {
        let (x1, x2) = sample();
        let permutation = [4, 1, 6, 0, 5, 2, 3];
        let mut scaled_x1 = [Vec2F64::ZERO; 7];
        let mut scaled_x2 = [Vec2F64::ZERO; 7];
        for (i, &source) in permutation.iter().enumerate() {
            scaled_x1[i] = Vec2F64::new(
                1_000.0 * x1[source].x + 350.0,
                1_000.0 * x1[source].y - 710.0,
            );
            scaled_x2[i] = Vec2F64::new(0.001 * x2[source].x - 0.4, 0.001 * x2[source].y + 0.8);
        }
        let models = fundamental_7point(&scaled_x1, &scaled_x2).unwrap();
        assert!(!models.is_empty());
        for model in models {
            for i in 0..7 {
                assert!(epipolar_error(&model, scaled_x1[i], scaled_x2[i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn polynomial_solver_handles_cubic_and_lower_degrees() {
        assert_eq!(real_polynomial_roots([1.0, -6.0, 11.0, -6.0]).1, 3);
        assert_eq!(real_polynomial_roots([1.0, -3.0, 3.0, -1.0]).1, 1);
        assert_eq!(real_polynomial_roots([0.0, 1.0, -3.0, 2.0]).1, 2);
        let (roots, count) = real_polynomial_roots([0.0, 0.0, 2.0, -4.0]);
        assert_eq!((&roots[..count]), &[2.0]);
    }

    #[test]
    fn polynomial_solver_preserves_close_and_tiny_roots() {
        let (roots, count) = real_polynomial_roots([0.0, 1.0, 0.0, -1e-16]);
        assert_eq!(count, 2);
        assert!(roots[..count]
            .iter()
            .any(|root| (*root - 1e-8).abs() < 1e-14));
        assert!(roots[..count]
            .iter()
            .any(|root| (*root + 1e-8).abs() < 1e-14));

        let coefficients = [3e-200, -6e-200, -9e-200, 18e-200];
        let (roots, count) = real_polynomial_roots(coefficients);
        assert_eq!(count, 3);
        for root in roots[..count].iter().copied() {
            let residual = ((coefficients[0] * root + coefficients[1]) * root + coefficients[2])
                * root
                + coefficients[3];
            assert!(residual.abs() <= 1e-212);
        }
    }

    #[test]
    fn degree_drop_includes_projective_infinity_candidate() {
        let f0 = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let f1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let coefficients = determinant_polynomial(&f0, &f1);
        assert_eq!(coefficients, [0.0, 3.0, 9.0, 6.0]);
        let (roots, count) = conditioned_roots(coefficients);
        assert_eq!(count, 3);
        let expected = frobenius_normalize::<false>(f1).unwrap();
        assert!(roots[..count].iter().any(|&(_, alpha, beta)| {
            let f = std::array::from_fn(|i| alpha * f0[i] + beta * f1[i]);
            let actual = frobenius_normalize::<false>(f).unwrap();
            matrices_proportional(&actual, &expected)
        }));
    }

    #[test]
    fn rank_one_candidate_is_not_a_fundamental_matrix() {
        let rank_one = [1.0, 2.0, 3.0, -2.0, -4.0, -6.0, 0.5, 1.0, 1.5];
        let rank_two = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!has_rank_at_least_two(
            &rank_one,
            vector_norm::<false>(&rank_one)
        ));
        assert!(has_rank_at_least_two(
            &rank_two,
            vector_norm::<false>(&rank_two)
        ));
    }

    #[test]
    fn random_nonplanar_samples_fit_every_candidate_and_a_holdout() {
        let mut seed = 0x82a3_9d5f_401b_c6e7;
        for trial in 0..16 {
            let (x1, x2) = random_sample(&mut seed, trial);
            let models = fundamental_7point(&x1[..7], &x2[..7]).unwrap();
            assert!((1..=3).contains(&models.len()));
            for model in &models {
                assert!(model.determinant().abs() < 1e-8);
                for i in 0..7 {
                    assert!(epipolar_error(model, x1[i], x2[i]).abs() < 1e-8);
                }
            }
            assert!(models
                .iter()
                .any(|model| epipolar_error(model, x1[7], x2[7]).abs() < 1e-8));
        }
    }

    #[test]
    fn singular_pencil_endpoints_keep_both_distinct_solutions() {
        // Kornia's singular-endpoint regression: full row rank, with one
        // simple and one double projective root.
        let x1 = [
            Vec2F64::new(0.0, -2.0),
            Vec2F64::new(-2.0, 1.0),
            Vec2F64::new(0.0, 1.0),
            Vec2F64::new(0.0, 0.0),
            Vec2F64::new(-1.0, 2.0),
            Vec2F64::new(-1.0, 1.0),
            Vec2F64::new(2.0, -1.0),
        ];
        let x2 = [
            Vec2F64::new(1.0, 2.0),
            Vec2F64::new(2.0, 2.0),
            Vec2F64::new(0.0, -2.0),
            Vec2F64::new(-1.0, 2.0),
            Vec2F64::new(0.0, -1.0),
            Vec2F64::new(-1.0, -2.0),
            Vec2F64::new(2.0, 2.0),
        ];
        let models = fundamental_7point(&x1, &x2).unwrap();
        assert!(models.len() >= 2, "missing repeated root: {models:?}");
        for expected in [
            [2.0, 4.0, 0.0, 0.0, -1.0, 0.0, -2.0, -2.0, 0.0],
            [12.0, 0.0, 0.0, -9.0, -1.0, 1.0, -6.0, 2.0, -2.0],
        ] {
            let expected = frobenius_normalize::<false>(expected).unwrap();
            assert!(models.iter().any(|model| {
                let a: [f64; 9] = (*model).into();
                let b: [f64; 9] = expected.into();
                let same = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
                let opposite = a.iter().zip(b).map(|(x, y)| (x + y).powi(2)).sum::<f64>();
                same.min(opposite) < 1e-16
            }));
        }
        for model in models {
            for i in 0..7 {
                assert!(epipolar_error(&model, x1[i], x2[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn invalid_and_degenerate_inputs_are_rejected() {
        let (x1, x2) = sample();
        assert!(fundamental_7point(&x1[..6], &x2[..6]).is_err());
        let mut non_finite = x1[..7].to_vec();
        non_finite[0].x = f64::NAN;
        assert!(fundamental_7point(&non_finite, &x2[..7]).is_err());
        let repeated = [Vec2F64::new(2.0, -1.0); 7];
        assert!(fundamental_7point(&repeated, &x2[..7]).is_err());
    }

    #[test]
    fn fixed_buffer_matches_public_wrapper_without_writing_past_results() {
        let (x1, x2) = sample();
        let expected = fundamental_7point(&x1[..7], &x2[..7]).unwrap();
        let sentinel = Mat3F64::IDENTITY;
        let mut models = [sentinel; 3];
        let count = fundamental_7point_into(&x1[..7], &x2[..7], &mut models).unwrap();

        assert_eq!(count, expected.len());
        for (actual, expected) in models[..count].iter().zip(expected) {
            assert!(matrices_proportional(actual, &expected));
        }
        assert!(models[count..].iter().all(|model| *model == sentinel));
    }

    #[test]
    fn real_sample_with_ill_conditioned_lu_gauge_keeps_rank_two() {
        // A real St Peter's sample has well-conditioned constraints but an
        // unstable LU basis when F21/F22 are fixed as the free coefficients.
        // Its legacy Cardano reconstruction leaves a normalized determinant
        // near 1e-8; adaptive pivoting plus an orthonormal pencil avoids that.
        let x1 = [
            Vec2F64::new(1.7077890204133563, 0.5580373472864585),
            Vec2F64::new(1.4107393432076643, -0.37911106559914304),
            Vec2F64::new(-0.4320745090186646, 0.010868851706561064),
            Vec2F64::new(-1.2231573171790784, -0.33701124450813225),
            Vec2F64::new(1.2125345919848851, -1.0107118458573),
            Vec2F64::new(-0.3602315414255174, 0.9753931451215252),
            Vec2F64::new(-2.3155995879826454, 0.18253481185003043),
        ];
        let x2 = [
            Vec2F64::new(-1.513616466259035, 0.4527689494200778),
            Vec2F64::new(1.9325793306543604, -0.4088141112236818),
            Vec2F64::new(0.27083098463077554, -0.0985210069317771),
            Vec2F64::new(1.2048531295127727, -0.37669815634220377),
            Vec2F64::new(-1.5208938125753597, -0.6308261393466781),
            Vec2F64::new(1.2203604676111275, 0.9547306376540731),
            Vec2F64::new(-1.5941136335746413, 0.10735982677018976),
        ];
        let models = fundamental_7point(&x1, &x2).unwrap();
        assert_eq!(models.len(), 3);
        for model in &models {
            assert!(model.determinant().abs() < 1e-12);
            for i in 0..7 {
                assert!(epipolar_error(model, x1[i], x2[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fast_norm_falls_back_for_extreme_magnitudes() {
        for scale in [1e-300, 1.0, 1e300] {
            let values = [scale; 9];
            let norm = vector_norm::<false>(&values);
            assert!(norm.is_finite());
            assert!((norm / scale - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn fixed_buffer_preserves_input_errors() {
        let (x1, x2) = sample();
        let mut models = [Mat3F64::ZERO; 3];
        assert!(matches!(
            fundamental_7point_into(&x1[..6], &x2[..6], &mut models),
            Err(FundamentalError::InvalidInput)
        ));
    }
}
