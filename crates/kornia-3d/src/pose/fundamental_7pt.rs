//! Minimal seven-point fundamental-matrix solver.

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

use crate::pose::fundamental::{apply_reflector_col, FundamentalError};

const RANK_EPS: f64 = 1e-12;
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
    if x1.len() != 7
        || x2.len() != 7
        || x1
            .iter()
            .chain(x2)
            .any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return Err(FundamentalError::InvalidInput);
    }

    let (x1n, t1) = hartley_normalize(x1).ok_or(FundamentalError::DegenerateConfiguration)?;
    let (x2n, t2) = hartley_normalize(x2).ok_or(FundamentalError::DegenerateConfiguration)?;
    let [f0, f1] = null_space_7x9(&x1n, &x2n).ok_or(FundamentalError::DegenerateConfiguration)?;

    let coefficients = determinant_polynomial(&f0, &f1);
    let coefficient_scale = coefficients
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if !coefficient_scale.is_finite() || coefficient_scale <= ROOT_EPS {
        return Err(FundamentalError::DegenerateConfiguration);
    }
    let (roots, root_count) = conditioned_roots(coefficients);
    let mut model_count = 0;
    for &(_, alpha, beta) in &roots[..root_count] {
        let f = std::array::from_fn(|i| alpha * f0[i] + beta * f1[i]);
        add_model(models, &mut model_count, f, &t1, &t2);
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
    let mut keyed = [(0.0, 0.0, 0.0); 3];
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
    keyed[..count].sort_by(|left, right| left.0.total_cmp(&right.0));
    (keyed, count)
}

fn hartley_normalize(points: &[Vec2F64]) -> Option<([Vec2F64; 7], Mat3F64)> {
    let mut cx = 0.0;
    let mut cy = 0.0;
    for point in points {
        cx += point.x;
        cy += point.y;
    }
    cx /= 7.0;
    cy /= 7.0;

    let mut sum_distance = 0.0;
    for point in points {
        let dx = point.x - cx;
        let dy = point.y - cy;
        sum_distance += (dx * dx + dy * dy).sqrt();
    }
    let mean_distance = sum_distance / 7.0;
    if !mean_distance.is_finite() || mean_distance <= f64::EPSILON {
        return None;
    }
    let scale = std::f64::consts::SQRT_2 / mean_distance;
    let mut normalized = [Vec2F64::ZERO; 7];
    for (dst, src) in normalized.iter_mut().zip(points) {
        *dst = Vec2F64::new((src.x - cx) * scale, (src.y - cy) * scale);
    }
    Some((
        normalized,
        Mat3F64::from_cols(
            Vec3F64::new(scale, 0.0, 0.0),
            Vec3F64::new(0.0, scale, 0.0),
            Vec3F64::new(-scale * cx, -scale * cy, 1.0),
        ),
    ))
}

/// Compute an orthonormal basis of the two-dimensional null space of A.
///
/// The input matrix is stored as A-transpose (nine rows by seven columns), so
/// seven Householder reflections reduce it directly and leave Q e7 and Q e8
/// as the two null vectors.
fn null_space_7x9(x1: &[Vec2F64; 7], x2: &[Vec2F64; 7]) -> Option<[[f64; 9]; 2]> {
    let mut at = [[0.0; 9]; 7];
    for i in 0..7 {
        let x = x1[i].x;
        let y = x1[i].y;
        let xp = x2[i].x;
        let yp = x2[i].y;
        at[i] = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1.0];
    }

    let mut reflectors = [[0.0; 9]; 7];
    let mut first_norm = 0.0;
    // The fixed minimal problem lets each reflector specialize its bounds.
    // LLVM can unroll the short SIMD loops instead of testing dynamic lengths.
    macro_rules! reduce {
        ($k:literal) => {
            reduce_column::<$k>(&mut at, &mut reflectors, &mut first_norm)?;
        };
    }
    reduce!(0);
    reduce!(1);
    reduce!(2);
    reduce!(3);
    reduce!(4);
    reduce!(5);
    reduce!(6);

    let mut basis = [[0.0; 9]; 2];
    basis[0][7] = 1.0;
    basis[1][8] = 1.0;
    macro_rules! reflect_basis {
        ($k:literal) => {
            apply_reflector(&mut basis[0], &reflectors[$k], $k);
            apply_reflector(&mut basis[1], &reflectors[$k], $k);
        };
    }
    reflect_basis!(6);
    reflect_basis!(5);
    reflect_basis!(4);
    reflect_basis!(3);
    reflect_basis!(2);
    reflect_basis!(1);
    reflect_basis!(0);
    Some(basis)
}

#[inline(always)]
fn reduce_column<const K: usize>(
    at: &mut [[f64; 9]; 7],
    reflectors: &mut [[f64; 9]; 7],
    first_norm: &mut f64,
) -> Option<()> {
    let mut u = [0.0; 9];
    let mut norm_sq = 0.0;
    for i in K..9 {
        u[i] = at[K][i];
        norm_sq += u[i] * u[i];
    }
    let norm = norm_sq.sqrt();
    if K == 0 {
        *first_norm = norm;
    }
    if !norm.is_finite() || norm <= RANK_EPS * first_norm.max(1.0) {
        return None;
    }
    let x0 = u[K];
    let alpha = if x0 >= 0.0 { -norm } else { norm };
    u[K] -= alpha;
    // Hartley normalization bounds the design matrix. The analytic norm
    // avoids rescaling and scanning the reflector a second time.
    let u_norm = (2.0 * norm * (norm + x0.abs())).sqrt();
    if !u_norm.is_finite() || u_norm <= f64::MIN_POSITIVE {
        return None;
    }
    let inv_norm = 1.0 / u_norm;
    for value in &mut u[K..] {
        *value *= inv_norm;
    }
    reflectors[K] = u;
    // The pivot column is never read again; only trailing columns need H.
    for column in at.iter_mut().skip(K + 1) {
        apply_reflector(column, &u, K);
    }
    Some(())
}

#[inline(always)]
fn apply_reflector(column: &mut [f64; 9], vector: &[f64; 9], start: usize) {
    apply_reflector_col(column, vector, start);
}

fn add_model(
    models: &mut [Mat3F64; 3],
    model_count: &mut usize,
    f: [f64; 9],
    t1: &Mat3F64,
    t2: &Mat3F64,
) {
    let fnorm = vector_norm(&f);
    if !fnorm.is_finite()
        || fnorm <= f64::MIN_POSITIVE
        || determinant(&f).abs() > 1e-8 * fnorm.powi(3)
        || !has_rank_at_least_two(&f, fnorm)
    {
        return;
    }
    let denormalized = denormalize(&f, t1, t2);
    let Some(result) = frobenius_normalize(denormalized) else {
        return;
    };
    if !models[..*model_count]
        .iter()
        .any(|other| matrices_proportional(other, &result))
    {
        models[*model_count] = result;
        *model_count += 1;
    }
}

/// Check that a rank-two candidate has a numerically nonzero two-by-two minor.
///
/// The determinant constraint alone also admits rank-one matrices, which do not
/// represent a valid fundamental matrix and must not reach RANSAC scoring.
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

fn determinant_columns(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]) -> f64 {
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
}

fn determinant(f: &[f64; 9]) -> f64 {
    f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6])
}

/// Return all distinct finite real roots of a polynomial of degree at most three.
fn real_polynomial_roots([a, b, c, d]: [f64; 4]) -> ([f64; 3], usize) {
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if !scale.is_finite() || scale == 0.0 {
        return ([0.0; 3], 0);
    }
    let [a, b, c, d] = [a / scale, b / scale, c / scale, d / scale];
    let epsilon = ROOT_EPS;
    let (mut roots, mut root_count) = if a.abs() <= epsilon {
        quadratic_roots(b, c, d, epsilon)
    } else {
        cubic_roots(a, b, c, d)
    };
    for root in &mut roots[..root_count] {
        for _ in 0..3 {
            let value = ((a * *root + b) * *root + c) * *root + d;
            // Once the Horner residual is at its rounding-error floor,
            // Newton can only amplify noise, particularly at a double root.
            let magnitude =
                ((a.abs() * root.abs() + b.abs()) * root.abs() + c.abs()) * root.abs() + d.abs();
            if value.abs() <= 4.0 * f64::EPSILON * magnitude {
                break;
            }
            let derivative = (3.0 * a * *root + 2.0 * b) * *root + c;
            if !value.is_finite() || !derivative.is_finite() || derivative.abs() <= ROOT_EPS {
                break;
            }
            let next = *root - value / derivative;
            if !next.is_finite() {
                break;
            }
            *root = next;
        }
    }
    let mut write = 0;
    for read in 0..root_count {
        if roots[read].is_finite() {
            roots[write] = roots[read];
            write += 1;
        }
    }
    root_count = write;
    roots[..root_count].sort_by(f64::total_cmp);
    write = 0;
    for read in 0..root_count {
        if write == 0
            || (roots[read] - roots[write - 1]).abs()
                > ROOT_EPS * roots[read].abs().max(roots[write - 1].abs()).max(1.0)
        {
            roots[write] = roots[read];
            write += 1;
        }
    }
    (roots, write)
}

fn quadratic_roots(a: f64, b: f64, c: f64, epsilon: f64) -> ([f64; 3], usize) {
    if a.abs() <= epsilon {
        return if b.abs() <= epsilon {
            ([0.0; 3], 0)
        } else {
            ([-c / b, 0.0, 0.0], 1)
        };
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
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

fn cubic_roots(a: f64, b: f64, c: f64, d: f64) -> ([f64; 3], usize) {
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    let offset = -b / (3.0 * a);
    let half_q = q * 0.5;
    let discriminant = half_q * half_q + (p / 3.0).powi(3);
    // Include rounding from the QR basis, mixed determinant coefficients and
    // projective reparameterization, as well as Cardano cancellation.
    let tolerance = 128.0 * f64::EPSILON * (half_q * half_q + (p / 3.0).abs().powi(3));
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
    let radius = 2.0 * (-p / 3.0).sqrt();
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
/// The sparse products preserve the multiplication order of the general path.
fn denormalize(f: &[f64; 9], t1: &Mat3F64, t2: &Mat3F64) -> Mat3F64 {
    let s1 = t1.x_axis.x;
    let s2 = t2.x_axis.x;
    let tx1 = t1.z_axis.x;
    let ty1 = t1.z_axis.y;
    let tx2 = t2.z_axis.x;
    let ty2 = t2.z_axis.y;
    let row0 = [s2 * f[0], s2 * f[1], s2 * f[2]];
    let row1 = [s2 * f[3], s2 * f[4], s2 * f[5]];
    let row2 = [
        tx2 * f[0] + ty2 * f[3] + f[6],
        tx2 * f[1] + ty2 * f[4] + f[7],
        tx2 * f[2] + ty2 * f[5] + f[8],
    ];
    Mat3F64::from_cols(
        Vec3F64::new(row0[0] * s1, row1[0] * s1, row2[0] * s1),
        Vec3F64::new(row0[1] * s1, row1[1] * s1, row2[1] * s1),
        Vec3F64::new(
            row0[0] * tx1 + row0[1] * ty1 + row0[2],
            row1[0] * tx1 + row1[1] * ty1 + row1[2],
            row2[0] * tx1 + row2[1] * ty1 + row2[2],
        ),
    )
}

#[cfg(test)]
fn vec9_to_mat3(f: &[f64; 9]) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(f[0], f[3], f[6]),
        Vec3F64::new(f[1], f[4], f[7]),
        Vec3F64::new(f[2], f[5], f[8]),
    )
}

fn vector_norm(values: &[f64; 9]) -> f64 {
    let squared = values.iter().map(|value| value * value).sum::<f64>();
    if squared.is_finite() && squared >= f64::MIN_POSITIVE {
        return squared.sqrt();
    }
    // Retain scaled arithmetic for extreme projective roots and coordinates.
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

fn frobenius_normalize(matrix: Mat3F64) -> Option<Mat3F64> {
    let flat: [f64; 9] = matrix.into();
    let norm = vector_norm(&flat);
    if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
        return None;
    }
    let mut normalized = flat;
    let inv_norm = 1.0 / norm;
    for value in &mut normalized {
        *value *= inv_norm;
    }
    Some(Mat3F64::from_cols(
        Vec3F64::new(normalized[0], normalized[1], normalized[2]),
        Vec3F64::new(normalized[3], normalized[4], normalized[5]),
        Vec3F64::new(normalized[6], normalized[7], normalized[8]),
    ))
}

fn matrices_proportional(left: &Mat3F64, right: &Mat3F64) -> bool {
    let left_flat: [f64; 9] = (*left).into();
    let right_flat: [f64; 9] = (*right).into();
    let same = left_flat
        .iter()
        .zip(right_flat)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let opposite = left_flat
        .iter()
        .zip(right_flat)
        .map(|(a, b)| (a + b).powi(2))
        .sum::<f64>();
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
        let expected = frobenius_normalize(vec9_to_mat3(&f1)).unwrap();
        assert!(roots[..count].iter().any(|&(_, alpha, beta)| {
            let f = std::array::from_fn(|i| alpha * f0[i] + beta * f1[i]);
            let actual = frobenius_normalize(vec9_to_mat3(&f)).unwrap();
            matrices_proportional(&actual, &expected)
        }));
    }

    #[test]
    fn rank_one_candidate_is_not_a_fundamental_matrix() {
        let rank_one = [1.0, 2.0, 3.0, -2.0, -4.0, -6.0, 0.5, 1.0, 1.5];
        let rank_two = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!has_rank_at_least_two(&rank_one, vector_norm(&rank_one)));
        assert!(has_rank_at_least_two(&rank_two, vector_norm(&rank_two)));
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
            let expected = frobenius_normalize(vec9_to_mat3(&expected)).unwrap();
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
        // unstable LU basis. Its legacy Cardano reconstruction leaves a
        // normalized determinant near 1e-8; orthonormal QR avoids cancellation.
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
            let norm = vector_norm(&values);
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
