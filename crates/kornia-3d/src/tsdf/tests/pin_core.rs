//! Pins of the shared-core numerics (contract sections 1-9). Each rule is
//! written once more here from the contract text with `libm` (not from the
//! code under test), compared bit for bit, and paired with the plausible
//! wrong variant: a witness count proves the inputs can tell them apart.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_range_contains,
    clippy::needless_range_loop
)]

use kornia_oxide_kernels::tsdf_core::RawDepth;

use super::*;
use crate::tsdf::alloc::{prepare, ObsKind};

/// `libm` twin of `StdOps`: the independent reference implementation.
struct LibmOps;

impl F32Ops for LibmOps {
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        libm::fmaf(a, b, c)
    }

    fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }
}

fn fma(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

/// Next float toward `+inf` (finite `x`).
fn succ(x: f32) -> f32 {
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

/// Next float toward `-inf` (finite `x`).
fn pred(x: f32) -> f32 {
    -succ(-x)
}

/// Uniform in `[lo, hi)`.
fn range(s: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * ((rng(s) >> 40) as f32 / 16_777_216.0)
}

fn bits3(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

/// The wrong variant must have told itself apart at least once.
fn witness(name: &str, differing: usize) {
    assert!(differing > 0, "{name}: no input separates it from the spec");
}

fn fnv(h: &mut u64, v: u32) {
    for b in v.to_le_bytes() {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

const ID: tc::Rigid = tc::Rigid {
    r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    t: [0.0; 3],
};

// ------------------------------------------------------------ transforms ----

#[test]
fn tsdf_xform_rot_op_order() {
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut w = [0usize; 6];
    for _ in 0..20_000 {
        let r: [f32; 9] = std::array::from_fn(|_| range(&mut s, -1.0, 1.0));
        let t: [f32; 3] = std::array::from_fn(|_| range(&mut s, -5.0, 5.0));
        let p: [f32; 3] = std::array::from_fn(|_| range(&mut s, -5.0, 5.0));
        // section 4: voxel x is the OUTERMOST fma
        let want: [f32; 3] = std::array::from_fn(|i| {
            fma(
                r[3 * i],
                p[0],
                fma(r[3 * i + 1], p[1], fma(r[3 * i + 2], p[2], t[i])),
            )
        });
        assert_eq!(
            bits3(tc::xform::<StdOps>(&r, &t, p)),
            bits3(want),
            "{r:?} {t:?} {p:?}"
        );
        let z_outer: [f32; 3] = std::array::from_fn(|i| {
            fma(
                r[3 * i + 2],
                p[2],
                fma(r[3 * i + 1], p[1], fma(r[3 * i], p[0], t[i])),
            )
        });
        let unfused: [f32; 3] = std::array::from_fn(|i| {
            r[3 * i] * p[0] + r[3 * i + 1] * p[1] + r[3 * i + 2] * p[2] + t[i]
        });
        w[0] += (bits3(z_outer) != bits3(want)) as usize;
        w[1] += (bits3(unfused) != bits3(want)) as usize;

        // rot: the last term is a plain product
        let want_rot: [f32; 3] = std::array::from_fn(|i| {
            fma(r[3 * i], p[0], fma(r[3 * i + 1], p[1], r[3 * i + 2] * p[2]))
        });
        assert_eq!(bits3(tc::rot::<StdOps>(&r, p)), bits3(want_rot));
        let rot_z_outer: [f32; 3] = std::array::from_fn(|i| {
            fma(r[3 * i + 2], p[2], fma(r[3 * i + 1], p[1], r[3 * i] * p[0]))
        });
        let rot_unfused: [f32; 3] =
            std::array::from_fn(|i| r[3 * i] * p[0] + r[3 * i + 1] * p[1] + r[3 * i + 2] * p[2]);
        w[2] += (bits3(rot_z_outer) != bits3(want_rot)) as usize;
        w[3] += (bits3(rot_unfused) != bits3(want_rot)) as usize;

        let want_n2 = fma(p[0], p[0], fma(p[1], p[1], p[2] * p[2]));
        assert_eq!(tc::norm2::<StdOps>(p).to_bits(), want_n2.to_bits());
        w[4] += ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).to_bits() != want_n2.to_bits()) as usize;
        let inv = 1.0 / libm::sqrtf(want_n2);
        let want_unit = [p[0] * inv, p[1] * inv, p[2] * inv];
        assert_eq!(bits3(tc::normalize::<StdOps>(p)), bits3(want_unit));
        let l = libm::sqrtf(want_n2);
        w[5] += (bits3([p[0] / l, p[1] / l, p[2] / l]) != bits3(want_unit)) as usize;
    }
    for (name, n) in [
        "xform z-outermost",
        "xform unfused",
        "rot z-outermost",
        "rot unfused",
        "norm2 unfused",
        "normalize by division",
    ]
    .iter()
    .zip(w)
    {
        witness(name, n);
    }
    // no division by zero: an underflowing square length normalises to zero
    assert_eq!(tc::normalize::<StdOps>([1e-23, 0.0, 0.0]), [0.0; 3]);
    assert_eq!(tc::normalize::<StdOps>([0.0; 3]), [0.0; 3]);
}

#[test]
fn tsdf_voxel_centre_matches_half_voxel_form() {
    let mut s = 99u64;
    let mut unfused = 0;
    for i in 0..50_000 {
        let vs = if i % 2 == 0 {
            range(&mut s, 1e-4, 10.0)
        } else {
            0.02 + 1e-4 * (i % 7) as f32
        };
        let half = vs * 0.5;
        let g = [
            ((rng(&mut s) >> 41) as i32) - (1 << 22),
            ((rng(&mut s) >> 41) as i32) - (1 << 22),
            ((rng(&mut s) >> 41) as i32) - (1 << 22),
        ];
        let c = tc::voxel_centre::<StdOps>(g, vs, half);
        for a in 0..3 {
            // (g + 0.5) * voxel_size, one rounding (|g| < 2^22: g + 0.5 is exact)
            let want = (g[a] as f32 + 0.5) * vs;
            assert_eq!(c[a].to_bits(), want.to_bits(), "g {g:?} vs {vs}");
            unfused += ((g[a] as f32 * vs + half).to_bits() != want.to_bits()) as usize;
        }
    }
    witness("g * vs + half unfused", unfused);
}

// --------------------------------------------------------- grid and keys ----

#[test]
fn tsdf_grid_domain_gates() {
    // inv_voxel = 1 makes q = x
    let at = |x: f32| tc::grid_coord(x, 1.0);
    assert_eq!(at(-4_194_304.0), Some(-4_194_304), "q == -2^22 is inside");
    assert_eq!(at(pred(-4_194_304.0)), None);
    assert_eq!(at(4_194_304.0 - 0.25), Some(4_194_303));
    assert_eq!(at(4_194_304.0), None, "q == 2^22 is outside");
    assert_eq!(at(-0.5), Some(-1), "floor, not truncation");
    assert_eq!(at(-0.0), Some(0));
    assert_eq!(at(0.999_999_94), Some(0));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert_eq!(at(bad), None);
    }
    let l = TsdfLayout::new(1.0, 4.0).unwrap();
    assert_eq!(
        l.voxel_from_point([-0.5, 0.5, -4_194_304.0]),
        Some([-1, 0, -4_194_304])
    );
    assert_eq!(l.voxel_from_point([0.0, 0.0, 4_194_304.0]), None);
}

#[test]
fn tsdf_touch_box_domain_and_floor() {
    let g = tc::Grid {
        voxel_size: 1.0,
        half_voxel: 0.5,
        inv_voxel: 1.0,
        trunc: 0.5,
        inv_trunc: 2.0,
        log2: 3,
    };
    // pv = 0: [-0.5, 0.5] is voxels -1..0 -> blocks -1..0 (truncation toward
    // zero would give 0..0)
    assert_eq!(tc::touch_box([0.0; 3], &g), Some(([-1; 3], [0; 3])));
    // lo == -2^22 exactly is accepted, one ulp lower is not
    assert_eq!(
        tc::touch_box([-4_194_303.5; 3], &g),
        Some(([-(1 << 19); 3], [-(1 << 19); 3]))
    );
    assert_eq!(tc::touch_box([-4_194_304.0, 0.0, 0.0], &g), None);
    // hi == 2^22 is rejected, the float below is accepted
    assert!(tc::touch_box([4_194_303.0; 3], &g).is_some());
    assert_eq!(tc::touch_box([0.0, 4_194_303.5, 0.0], &g), None);
    assert_eq!(tc::touch_box([0.0, 0.0, f32::NAN], &g), None);
    // the box is per axis: x spans a block boundary, y and z do not
    let b = tc::touch_box([7.9, 0.2, 0.2], &g).unwrap();
    assert_eq!(b, ([0, -1, -1], [1, 0, 0]));
    let g16 = tc::Grid { log2: 4, ..g };
    assert_eq!(
        tc::touch_box([7.9, 0.2, 0.2], &g16).unwrap(),
        ([0, -1, -1], [0, 0, 0])
    );
}

#[test]
fn tsdf_key_packing_golden() -> R {
    let hi = ((1u64 << 20) << 42) | ((1u64 << 20) << 21) | (1u64 << 20);
    assert_eq!(tc::pack_block_key([0, 0, 0]), hi);
    assert_eq!(tc::pack_block_key([-(1 << 20); 3]), 0);
    let top = tc::pack_block_key([(1 << 20) - 1; 3]);
    assert_eq!(top, (1u64 << 63) - 1);
    assert!(top < tc::EMPTY_KEY && top >> 63 == 0, "bit 63 is never set");
    assert_eq!(
        tc::pack_block_key([1, 0, 0]) - tc::pack_block_key([0, 0, 0]),
        1 << 42
    );
    assert_eq!(
        tc::pack_block_key([0, 0, 1]) - tc::pack_block_key([0, 0, 0]),
        1
    );
    // ascending keys are lexicographic signed (x, y, z); round trip
    let mut s = 5u64;
    let mut v: Vec<[i32; 3]> = (0..2000)
        .map(|_| std::array::from_fn(|_| ((rng(&mut s) >> 43) as i32) - (1 << 20)))
        .collect();
    v.push([-(1 << 20), 0, 7]);
    v.push([(1 << 20) - 1, -1, -(1 << 20)]);
    let mut by_key = v.clone();
    by_key.sort_by_key(|&b| tc::pack_block_key(b));
    v.sort();
    assert_eq!(by_key, v);
    for b in &v {
        assert_eq!(tc::unpack_block_key(tc::pack_block_key(*b)), *b);
    }
    // every axis rejects both ends of the key range
    for a in 0..3 {
        for bad in [-(1 << 20) - 1, 1 << 20] {
            let mut c = [0i32; 3];
            c[a] = bad;
            let e = BlockKey::new(c[0], c[1], c[2]).unwrap_err();
            assert!(
                matches!(e, TsdfError::KeyOutOfRange { x, y, z } if [x, y, z] == c),
                "{e}"
            );
        }
        for ok in [-(1 << 20), (1 << 20) - 1] {
            let mut c = [0i32; 3];
            c[a] = ok;
            assert_eq!(BlockKey::new(c[0], c[1], c[2])?.coords(), c);
        }
    }
    Ok(())
}

#[test]
fn tsdf_hash_lookup_equals_map() {
    for log2 in [2u32, 6] {
        let cap = 1usize << log2;
        // keys whose first probe slot is the last slot: chains wrap around
        let mut chain: Vec<u64> = (0u64..)
            .map(|i| tc::pack_block_key([i as i32, 0, 0]))
            .filter(|&k| tc::hash_slot(k, log2) == cap - 1)
            .take(cap / 2 + 1)
            .collect();
        let absent = chain.pop().unwrap();
        let mut keys = vec![tc::EMPTY_KEY; cap];
        let mut vals = vec![0u32; cap];
        for (n, &k) in chain.iter().enumerate() {
            assert!(tc::hash_insert(
                &mut keys,
                &mut vals,
                log2,
                k,
                100 + n as u32
            ));
        }
        for (n, &k) in chain.iter().enumerate() {
            assert_eq!(tc::hash_lookup(&keys, &vals, log2, k), Some(100 + n as u32));
        }
        assert_eq!(tc::hash_lookup(&keys, &vals, log2, absent), None);
    }
    // a full table: insert fails, an absent key still terminates
    let mut keys = vec![tc::EMPTY_KEY; 4];
    let mut vals = vec![0u32; 4];
    for i in 0..4 {
        assert!(tc::hash_insert(&mut keys, &mut vals, 2, i, i as u32));
    }
    assert!(!tc::hash_insert(&mut keys, &mut vals, 2, 9, 9));
    assert_eq!(tc::hash_lookup(&keys, &vals, 2, 9), None);
    assert_eq!(tc::hash_lookup(&keys, &vals, 2, 3), Some(3));
}

// ---------------------------------------------------------------- depth ----

#[test]
fn tsdf_depth_sample_gates() -> R {
    // meter 2 -> mpu 0.5, so raw 2 is exactly 1.0 and raw 14 exactly 7.0
    let mpu = DepthImage::from_size_val(
        ImageSize {
            width: 1,
            height: 1,
        },
        0u16,
        2.0,
    )?
    .metres_per_unit();
    assert_eq!(mpu, 0.5);
    let u = |raw: u16| raw.metres(mpu, 1.0, 7.0);
    assert_eq!(u(0), None, "0 is the missing sample");
    assert_eq!(u(1), None);
    assert_eq!(u(2), Some(1.0), "range.min is kept");
    assert_eq!(u(13), Some(6.5));
    assert_eq!(u(14), None, "range.max is excluded");
    assert_eq!(u(u16::MAX), None);
    // f32 samples are metres * mpu; non-finite, non-positive and out of range drop
    let f = |raw: f32, m: f32| raw.metres(m, 1.0, 7.0);
    assert_eq!(f(1.0, 1.0), Some(1.0));
    assert_eq!(f(pred(1.0), 1.0), None);
    assert_eq!(f(7.0, 1.0), None);
    assert_eq!(f(pred(7.0), 1.0), Some(pred(7.0)));
    for bad in [
        0.0,
        -0.0,
        -1.0,
        1e-40,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(f(bad, 1.0), None, "{bad:e}");
    }
    assert_eq!(f(f32::MAX, 2.0), None, "overflow to inf is not a sample");
    assert_eq!(f(2.0, 0.5), Some(1.0));
    // mpu is 1 / meter in f32 and the sample is multiplied, never divided
    let mpu = 1.0f32 / 1000.0;
    let (mut divided, mut checked) = (0, 0);
    for raw in 1..=u16::MAX {
        let d = raw.metres(mpu, 1e-6, 1e6).unwrap();
        assert_eq!(d.to_bits(), (raw as f32 * mpu).to_bits());
        divided += ((raw as f32 / 1000.0).to_bits() != d.to_bits()) as usize;
        checked += 1;
    }
    assert_eq!(checked, 65535);
    witness("raw / meter", divided);
    for meter in [1000.0f32, 6553.5, 1.0] {
        let d = DepthImage::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            1u16,
            meter,
        )?;
        assert_eq!(d.metres_per_unit().to_bits(), (1.0f32 / meter).to_bits());
    }
    Ok(())
}

// ------------------------------------------------------ pinhole projection ----

fn unit_pinhole(w: u32, h: u32) -> tc::PinholeParams {
    tc::PinholeParams {
        sensor_from_volume: ID,
        volume_from_sensor: ID,
        fx: 1.0,
        fy: 1.0,
        cx: 0.0,
        cy: 0.0,
        inv_fx: 1.0,
        inv_fy: 1.0,
        width: w,
        height: h,
        width_f: w as f32,
        height_f: h as f32,
        mpu: 1.0,
        range_min: 0.5,
        range_max: 7.0,
    }
}

#[test]
fn tsdf_pinhole_pixel_rule_and_cheirality() {
    let p = unit_pinhole(8, 4);
    // fx = 1, cx = 0, z = 1: u = x, v = y; pixel = floor(coord + 0.5)
    let px = |x: f32, y: f32, z: f32| tc::pinhole_pixel::<StdOps>(&p, [x, y, z]);
    assert_eq!(px(-0.5, 0.0, 1.0), Some(0), "coord -0.5 is still pixel 0");
    assert_eq!(px(pred(-0.5), 0.0, 1.0), None);
    assert_eq!(px(7.5, 0.0, 1.0), None, "coord W - 0.5 is outside");
    assert_eq!(px(pred(7.5), 0.0, 1.0), Some(7));
    assert_eq!(
        px(3.5, 0.0, 1.0),
        Some(4),
        "round half up (floor(u) gives 3)"
    );
    assert_eq!(px(pred(3.5), 0.0, 1.0), Some(3));
    assert_eq!(px(2.9, 0.0, 1.0), Some(3));
    assert_eq!(px(0.0, 3.5, 1.0), None, "coord H - 0.5 is outside");
    assert_eq!(px(0.0, 2.5, 1.0), Some(3 * 8));
    assert_eq!(px(5.0, 2.0, 1.0), Some(2 * 8 + 5), "idx = py * W + px");
    // cheirality: z >= range.min, NaN-safe
    assert_eq!(px(0.0, 0.0, 0.5), Some(0));
    assert_eq!(px(0.0, 0.0, pred(0.5)), None);
    for z in [0.0, -1.0, f32::NAN, -0.0] {
        assert_eq!(px(0.0, 0.0, z), None, "{z}");
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(px(bad, 0.0, 1.0), None);
        assert_eq!(px(0.0, bad, 1.0), None);
    }
}

/// The contract's pinhole projection, transcribed with `libm`.
fn pinhole_spec(p: &tc::PinholeParams, s: [f32; 3]) -> Option<u32> {
    if !(s[2] >= p.range_min) {
        return None;
    }
    let iz = 1.0 / s[2];
    let u = fma(p.fx, s[0] * iz, p.cx);
    let v = fma(p.fy, s[1] * iz, p.cy);
    let (uh, vh) = (u + 0.5, v + 0.5);
    if !(uh >= 0.0 && uh < p.width_f && vh >= 0.0 && vh < p.height_f) {
        return None;
    }
    Some((vh as i32 as u32) * p.width + (uh as i32 as u32))
}

#[test]
fn tsdf_pinhole_projection_op_order() {
    let p = tc::PinholeParams {
        fx: 517.3,
        fy: 515.9,
        cx: 319.5,
        cy: 239.5,
        inv_fx: 1.0 / 517.3,
        inv_fy: 1.0 / 515.9,
        width: 640,
        height: 480,
        width_f: 640.0,
        height_f: 480.0,
        ..unit_pinhole(640, 480)
    };
    let mut s = 31u64;
    let (mut unfused, mut divided, mut hits) = (0, 0, 0);
    for _ in 0..2_000_000 {
        let z = range(&mut s, 0.5, 6.0);
        let q = [
            range(&mut s, -0.62, 0.62) * z,
            range(&mut s, -0.46, 0.46) * z,
            z,
        ];
        let got = tc::pinhole_pixel::<StdOps>(&p, q);
        assert_eq!(got, pinhole_spec(&p, q), "{q:?}");
        hits += got.is_some() as usize;
        let iz = 1.0 / z;
        let pix = |u: f32, v: f32| {
            let (uh, vh) = (u + 0.5, v + 0.5);
            (uh >= 0.0 && uh < 640.0 && vh >= 0.0 && vh < 480.0)
                .then(|| (vh as i32 as u32) * 640 + uh as i32 as u32)
        };
        unfused += (pix(p.fx * (q[0] * iz) + p.cx, p.fy * (q[1] * iz) + p.cy) != got) as usize;
        divided += (pix(p.fx * q[0] / z + p.cx, p.fy * q[1] / z + p.cy) != got) as usize;
    }
    assert!(hits > 1_000_000);
    witness("projection unfused", unfused);
    witness("projection fx*x/z", divided);
}

#[test]
fn tsdf_pinhole_unprojection_op_order() {
    let p = tc::PinholeParams {
        cx: 319.5,
        cy: 239.5,
        inv_fx: (1.0f64 / 517.3) as f32,
        inv_fy: (1.0f64 / 515.9) as f32,
        ..unit_pinhole(640, 480)
    };
    let (mut order, mut div) = (0, 0);
    let mut s = 8u64;
    for _ in 0..20_000 {
        let (x, y) = ((rng(&mut s) % 640) as u32, (rng(&mut s) % 480) as u32);
        let d = range(&mut s, 0.1, 7.0);
        let got = tc::pinhole_point(&p, x, y, d);
        // section 6: xn = (px - cx) * inv_fx, then xn * d
        let (xn, yn) = ((x as f32 - p.cx) * p.inv_fx, (y as f32 - p.cy) * p.inv_fy);
        assert_eq!(bits3(got), bits3([xn * d, yn * d, d]));
        order += (got[0].to_bits() != ((x as f32 - p.cx) * d * p.inv_fx).to_bits()) as usize;
        div += (got[0].to_bits() != ((x as f32 - p.cx) / 517.3f32 * d).to_bits()) as usize;
    }
    witness("(px - cx) * d * inv_fx", order);
    witness("(px - cx) / fx * d", div);
}

#[test]
fn tsdf_pinhole_sdf_is_projective() {
    // sdf = d - z (not the euclidean distance), with the pixel index
    let p = unit_pinhole(8, 4);
    let mut depth = vec![0.0f32; 32];
    depth[2 * 8 + 5] = 2.0;
    depth[3] = f32::NAN;
    let (sdf, idx) = tc::pinhole_sdf::<StdOps, f32>(&p, &depth, [5.0, 2.0, 1.0]).unwrap();
    assert_eq!(idx, 21);
    assert_eq!(sdf.to_bits(), (2.0f32 - 1.0).to_bits());
    // missing sample, NaN sample and a voxel outside the image drop
    assert!(tc::pinhole_sdf::<StdOps, f32>(&p, &depth, [1.0, 1.0, 1.5]).is_none());
    assert!(tc::pinhole_sdf::<StdOps, f32>(&p, &depth, [3.0, 0.0, 1.0]).is_none());
    assert!(tc::pinhole_sdf::<StdOps, f32>(&p, &depth, [20.0, 0.0, 1.0]).is_none());
}

// --------------------------------------------------------------- spherical ----

fn sph(width: u32, height: u32, wrap: bool, fov: [f32; 2], az: [f32; 2]) -> tc::SphericalParams {
    tc::SphericalParams {
        sensor_from_volume: ID,
        volume_from_sensor: ID,
        fov_low: fov[0],
        fov_high: fov[1],
        inv_az_step: az[0],
        az_offset: az[1],
        width,
        height,
        wrap: wrap as u32,
        mpu: 1.0,
        range_min: 1.0,
        range_max: 100.0,
    }
}

#[test]
fn tsdf_spherical_row_and_range_gates() {
    // (3, 0, 4): r = 5 exactly, sz = RN(4 / 5)
    let s = [3.0f32, 0.0, 4.0];
    let sz = 4.0f32 / 5.0;
    let wide = [-0.5, 0.95];
    // count_le counts a boundary equal to sz (<=)
    let bnd = [0.5, sz, 0.9];
    assert_eq!(tc::count_le(&bnd, sz), 2);
    assert_eq!(tc::count_le(&[0.5, succ(sz), 0.9], sz), 1);
    assert_eq!(tc::count_le(&bnd, pred(0.5)), 0);
    assert_eq!(tc::count_le(&bnd, 0.9), 3);
    assert_eq!(tc::count_le(&[], 0.0), 0);
    let p = sph(8, 4, true, wide, [1.0, 0.0]);
    let (r, idx) = tc::spherical_pixel::<StdOps>(&p, &bnd, s).unwrap();
    assert_eq!(r, 5.0);
    assert_eq!(idx, 8, "row = (H - 1) - 2 = 1, column 0");
    let (_, idx) = tc::spherical_pixel::<StdOps>(&p, &[0.5, succ(sz), 0.9], s).unwrap();
    assert_eq!(idx, 2 * 8);
    // vertical FOV: low is inclusive, high is exclusive
    let at = |lo: f32, hi: f32| {
        tc::spherical_pixel::<StdOps>(&sph(8, 4, true, [lo, hi], [1.0, 0.0]), &bnd, s)
    };
    assert!(at(sz, 0.95).is_some());
    assert!(at(succ(sz), 0.95).is_none());
    assert!(at(0.0, sz).is_none());
    assert!(at(0.0, succ(sz)).is_some());
    // range gate before anything else: r >= range_min
    let mut q = p;
    q.range_min = 5.0;
    assert!(tc::spherical_pixel::<StdOps>(&q, &bnd, s).is_some());
    q.range_min = succ(5.0);
    assert!(tc::spherical_pixel::<StdOps>(&q, &bnd, s).is_none());
    for bad in [[0.0; 3], [f32::NAN, 0.0, 4.0]] {
        assert!(
            tc::spherical_pixel::<StdOps>(&p, &bnd, bad).is_none(),
            "{bad:?}"
        );
    }
}

#[test]
fn tsdf_spherical_column_rule() {
    let bnd = [-0.5, 0.0, 0.5];
    let fov = [-0.75, 0.75];
    // 1 column per radian; z = 0 so sz = 0 and the row is 1
    let w8 = sph(8, 4, true, fov, [1.0, 0.0]);
    let cw = |p: &tc::SphericalParams, s: [f32; 3]| {
        tc::spherical_pixel::<StdOps>(p, &bnd, s).map(|(_, i)| i)
    };
    // phi = -pi/4: coord -0.285 -> ci -1 -> wraps to column 7 (a Rust `%` gives -1)
    assert_eq!(cw(&w8, [1.0, -1.0, 0.0]), Some(8 + 7));
    assert_eq!(
        cw(&sph(8, 4, false, fov, [1.0, 0.0]), [1.0, -1.0, 0.0]),
        None
    );
    // phi = pi: coord 3.64 -> column 3; with 3 columns ci == W wraps / is rejected
    assert_eq!(cw(&w8, [-1.0, 0.0, 0.0]), Some(8 + 3));
    assert_eq!(
        cw(&sph(3, 4, true, fov, [1.0, 0.0]), [-1.0, 0.0, 0.0]),
        Some(3)
    );
    assert_eq!(
        cw(&sph(3, 4, false, fov, [1.0, 0.0]), [-1.0, 0.0, 0.0]),
        None
    );
    // atan2_det(-0, x < 0) = +pi: both signs of zero land in the same column
    assert_eq!(cw(&w8, [-1.0, -0.0, 0.0]), cw(&w8, [-1.0, 0.0, 0.0]));
    // a negative step (clockwise spinner): coord = phi * inv + offset
    assert_eq!(
        cw(&sph(8, 4, true, fov, [-1.0, 2.0]), [1.0, 1.0, 0.0]),
        Some(8 + 1)
    );
    // |coord + 0.5| >= 2^24 is rejected before any integer conversion, even when wrapping
    for y in [1.0f32, -1.0] {
        assert_eq!(cw(&sph(8, 4, true, fov, [1.0e8, 0.0]), [1.0, y, 0.0]), None);
    }
    // large but in-range coordinates wrap with rem_euclid
    let p = sph(8, 4, true, fov, [1.0e6, 0.0]);
    for y in [1.0f32, -1.0] {
        let phi = tc::atan2_det::<LibmOps>(y, 1.0);
        let ci = (fma(phi, 1.0e6, 0.0) + 0.5).floor() as i32;
        assert_eq!(cw(&p, [1.0, y, 0.0]), Some(8 + ci.rem_euclid(8) as u32));
    }
}

#[test]
fn tsdf_spherical_sdf_is_range_space() {
    let p = sph(8, 4, true, [-0.75, 0.75], [1.0, 0.0]);
    let bnd = [-0.5, 0.0, 0.5];
    let mut range = vec![0.0f32; 32];
    range[8] = 6.25;
    let (sdf, idx) = tc::spherical_sdf::<StdOps, f32>(&p, &bnd, &range, [3.0, 0.0, 0.0]).unwrap();
    assert_eq!(idx, 8);
    assert_eq!(
        sdf.to_bits(),
        (6.25f32 - 3.0).to_bits(),
        "rho - |p|, not a z difference"
    );
    // an empty range pixel drops the voxel
    assert!(tc::spherical_sdf::<StdOps, f32>(&p, &bnd, &range, [-3.0, 0.0, 0.0]).is_none());
}

/// Section 7.3 transcribed with `libm`.
fn spherical_spec(p: &tc::SphericalParams, bnd: &[f32], s: [f32; 3]) -> Option<(f32, u32)> {
    let rr = fma(s[0], s[0], fma(s[1], s[1], s[2] * s[2]));
    let r = libm::sqrtf(rr);
    if !(r >= p.range_min) {
        return None;
    }
    let sz = s[2] / r;
    if !(p.fov_low <= sz && sz < p.fov_high) {
        return None;
    }
    let k = bnd.iter().filter(|&&b| b <= sz).count() as u32;
    let row = (p.height - 1) - k;
    let phi = tc::atan2_det::<LibmOps>(s[1], s[0]);
    let uh = fma(phi, p.inv_az_step, p.az_offset) + 0.5;
    if !(uh >= -16_777_216.0 && uh < 16_777_216.0) {
        return None;
    }
    let ci = uh.floor() as i32;
    let w = p.width as i32;
    let col = if p.wrap != 0 {
        ci.rem_euclid(w)
    } else if (0..w).contains(&ci) {
        ci
    } else {
        return None;
    };
    Some((r, row * p.width + col as u32))
}

#[test]
fn tsdf_spherical_pixel_matches_spec() {
    let l = SphericalLidar::uniform(
        ImageSize {
            width: 360,
            height: 16,
        },
        AzimuthGrid {
            start: 0.0,
            step: std::f64::consts::TAU / 360.0,
            wrap: true,
        },
        0.3,
        -0.3,
        DepthRange::new(0.2, 50.0).unwrap(),
    )
    .unwrap();
    let t = l.tables();
    let p = l.params(ID, ID, 1.0);
    let mut s = 77u64;
    let (mut unfused, mut hits) = (0, 0);
    for _ in 0..300_000 {
        let q = [
            range(&mut s, -20.0, 20.0),
            range(&mut s, -20.0, 20.0),
            range(&mut s, -6.0, 6.0),
        ];
        let got = tc::spherical_pixel::<StdOps>(&p, t.row_boundaries, q);
        let want = spherical_spec(&p, t.row_boundaries, q);
        assert_eq!(
            got.map(|(r, i)| (r.to_bits(), i)),
            want.map(|(r, i)| (r.to_bits(), i)),
            "{q:?}"
        );
        if let Some((r, _)) = got {
            hits += 1;
            let plain = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            unfused += (plain.to_bits() != r.to_bits()) as usize;
        }
    }
    assert!(hits > 20_000, "{hits}");
    witness("r from unfused squares", unfused);
}

// -------------------------------------------------------------- atan2_det ----

#[test]
fn tsdf_atan2_det_golden_bits() {
    let a = |y: f32, x: f32| tc::atan2_det::<StdOps>(y, x);
    let pi = f32::from_bits(0x4049_0fdb);
    let half_pi = f32::from_bits(0x3fc9_0fdb);
    // signed zeros: libm gives -0 and -pi here
    assert_eq!(a(-0.0, 1.0).to_bits(), 0, "(-0, 1) is +0");
    assert_eq!(a(0.0, 1.0).to_bits(), 0);
    assert_eq!(a(0.0, -1.0).to_bits(), pi.to_bits());
    assert_eq!(a(-0.0, -1.0).to_bits(), pi.to_bits(), "(-0, x<0) is +pi");
    assert_eq!(a(-0.0, -0.0).to_bits(), 0);
    assert_eq!(a(0.0, 0.0).to_bits(), 0);
    assert_eq!(a(0.0, -0.0).to_bits(), 0);
    assert_eq!(a(1.0, 0.0).to_bits(), half_pi.to_bits());
    assert_eq!(a(-1.0, -0.0).to_bits(), (-half_pi).to_bits());
    assert_eq!(a(-1.0, 0.0).to_bits(), (-half_pi).to_bits());
    assert_eq!(a(1e-30, 1.0).to_bits(), 1e-30f32.to_bits());
    // diagonals: the octant mirrors are exact
    let q = a(1.0, 1.0);
    assert_eq!(q.to_bits(), ATAN2_DIAGONAL, "atan2_det(1, 1)");
    assert_eq!(a(-1.0, -1.0).to_bits(), (-(pi - q)).to_bits());
    assert_eq!(a(1.0, -1.0).to_bits(), (pi - q).to_bits());
    assert_eq!(a(-1.0, 1.0).to_bits(), (-q).to_bits());
    // |y| > |x| mirrors about pi/2
    let t = a(0.5, 1.0);
    assert_eq!(a(1.0, 0.5).to_bits(), (half_pi - t).to_bits());
    // finite extremes and subnormals stay finite and exact in the octant
    assert_eq!(a(f32::MAX, f32::MAX).to_bits(), q.to_bits());
    assert_eq!(a(1e-45, 1e-45).to_bits(), q.to_bits());
    assert_eq!(
        a(1e-45, 1.0).to_bits(),
        1e-45f32.to_bits(),
        "no flush to zero"
    );
}

/// Bits of `atan2_det(1, 1)`.
const ATAN2_DIAGONAL: u32 = 0x3f49_0fda;

#[test]
fn tsdf_atan2_det_sweep_hash() {
    // fixed sweeps, independent of the build profile
    let mut h = FNV_OFFSET;
    let mut bits = 0u32;
    while bits <= 1.0f32.to_bits() {
        fnv(
            &mut h,
            tc::atan2_det::<StdOps>(f32::from_bits(bits), 1.0).to_bits(),
        );
        bits += 4093;
    }
    let mut s = 0x0123_4567_89ab_cdefu64;
    for _ in 0..100_000 {
        let v = rng(&mut s);
        let (y, x) = (f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32));
        if y.is_finite() && x.is_finite() {
            fnv(&mut h, tc::atan2_det::<StdOps>(y, x).to_bits());
        }
    }
    for i in 0..20_000 {
        let (y, x) = (range(&mut s, -3.0, 3.0), range(&mut s, -3.0, 3.0));
        fnv(&mut h, tc::atan2_det::<StdOps>(y, x).to_bits());
        let a = tc::atan2_det::<StdOps>(y, x);
        assert_eq!(a.to_bits(), tc::atan2_det::<LibmOps>(y, x).to_bits(), "{i}");
    }
    assert_eq!(h, ATAN2_SWEEP_HASH, "got {h:#x}");
}

const ATAN2_SWEEP_HASH: u64 = 0x2a12_269a_f4e9_0001;

// ------------------------------------------------------------ truncation ----

#[test]
fn tsdf_truncation_rules() {
    // RN(trunc * RN(1 / trunc)) is never above 1, but the clamp is part of the
    // contract for any inv_trunc: probe it with the neighbours of 1 / trunc too
    let (mut above, mut below, mut exact) = (0, 0, 0);
    let (mut clamp_needed, mut divided, mut free_not_one, mut unclamped) = (0, 0, 0, 0);
    for i in 1..4000u32 {
        let trunc = i as f32 * 0.00137;
        let inv0 = 1.0f32 / trunc;
        for inv in [pred(inv0), inv0, succ(inv0)] {
            let pr = trunc * inv;
            match pr.partial_cmp(&1.0) {
                Some(std::cmp::Ordering::Greater) => above += 1,
                Some(std::cmp::Ordering::Less) => below += 1,
                _ => exact += 1,
            }
            // behind the band: -trunc is kept, one ulp lower is not
            let want_edge = if -pr < -1.0 { -1.0 } else { -pr };
            assert_eq!(
                tc::truncate(-trunc, trunc, inv).map(f32::to_bits),
                Some(want_edge.to_bits()),
                "trunc {trunc} inv {inv}"
            );
            clamp_needed += (-pr < -1.0) as usize;
            unclamped += (-pr != want_edge) as usize;
            assert_eq!(tc::truncate(pred(-trunc), trunc, inv), None);
            // free space is exactly 1.0, even where sdf * inv_trunc is not
            assert_eq!(tc::truncate(trunc, trunc, inv), Some(1.0));
            assert_eq!(tc::truncate(succ(trunc), trunc, inv), Some(1.0));
            assert_eq!(tc::truncate(1e6, trunc, inv), Some(1.0));
            free_not_one += (pr != 1.0) as usize;
            // the upper clamp: just below trunc the product may round above 1
            let near = pred(trunc);
            let v = tc::truncate(near, trunc, inv).unwrap();
            assert!((-1.0..=1.0).contains(&v));
            let raw = near * inv;
            assert_eq!(v.to_bits(), if raw > 1.0 { 1.0f32 } else { raw }.to_bits());
        }
        // in the band the value is sdf * inv_trunc
        let sdf = 0.37 * trunc;
        let got = tc::truncate(sdf, trunc, inv0).unwrap();
        assert_eq!(got.to_bits(), (sdf * inv0).to_bits());
        divided += ((sdf / trunc).to_bits() != got.to_bits()) as usize;
    }
    assert!(
        above > 0 && below > 0 && exact > 0,
        "{above} {below} {exact}"
    );
    witness("clamp", clamp_needed);
    witness("sdf / trunc", divided);
    witness("free space as sdf * inv_trunc", free_not_one);
    assert!(unclamped > 0);
    assert_eq!(tc::truncate(f32::NAN, 1.0, 1.0), None);
    assert_eq!(tc::truncate(f32::NEG_INFINITY, 1.0, 1.0), None);
}

// ---------------------------------------------------------------- update ----

#[test]
fn tsdf_weighted_mean_rules() {
    let mut s = 4u64;
    let (mut ratio, mut capped_n, mut moved) = (0, 0, 0);
    for _ in 0..200_000 {
        let old = range(&mut s, -1.0, 1.0);
        let tsd = range(&mut s, -1.0, 1.0);
        let max_w = [3u16, 16, 255, u16::MAX][(rng(&mut s) % 4) as usize];
        let w0 = (rng(&mut s) % (max_w as u64 + 1)) as u16;
        let (t, w) = tc::update_tsdf::<StdOps>(old, w0, tsd, max_w);
        let n = w0 as u32 + 1;
        let want = fma(tsd - old, 1.0 / (n as f32), old);
        assert_eq!(t.to_bits(), want.to_bits());
        assert_eq!(w as u32, n.min(max_w as u32));
        ratio += ((old * w0 as f32 + tsd) / n as f32 != want) as usize;
        // at the cap the count that drives the mean is still w0 + 1
        if w0 == max_w {
            capped_n +=
                (fma(tsd - old, 1.0 / (max_w as f32), old).to_bits() != want.to_bits()) as usize;
        }
        moved += (t != old) as usize;
    }
    witness("(old * w + tsd) / (w + 1)", ratio);
    witness("capped count", capped_n);
    assert!(moved > 100_000);
    // the cap: old 1, tsd -1, w 3 of 3 -> 1 + (-2) / 4 = 0.5 (a 1/3 mean gives 0.333)
    let (t, w) = tc::update_tsdf::<StdOps>(1.0, 3, -1.0, 3);
    assert_eq!((t, w), (0.5, 3));
    // no u16 wrap at 65535
    let (t, w) = tc::update_tsdf::<StdOps>(0.5, u16::MAX, 1.0, u16::MAX);
    assert_eq!(w, u16::MAX);
    assert_eq!(t.to_bits(), fma(0.5, 1.0 / 65536.0, 0.5).to_bits());
    // first observation keeps tsd exactly
    for tsd in [0.3f32, -1.0, 1.0, 0.123_456_79] {
        assert_eq!(
            tc::update_tsdf::<StdOps>(0.0, 0, tsd, 9).0.to_bits(),
            tsd.to_bits()
        );
    }
}

#[test]
fn tsdf_signed_zero_rules() -> R {
    // a measurement exactly on the surface is truncated to -0.0 ...
    let tsd = tc::truncate(-0.0, 0.08, 12.5).unwrap();
    assert_eq!(tsd.to_bits(), 0x8000_0000);
    // ... and the first update stores +0 (copying tsd would store -0)
    let (t, w) = tc::update_tsdf::<StdOps>(0.0, 0, tsd, 10);
    assert_eq!((t.to_bits(), w), (0, 1));
    // exact cancellation rounds to +0 as well
    assert_eq!(tc::update_tsdf::<StdOps>(0.5, 1, -0.5, 10).0.to_bits(), 0);
    assert_eq!(tc::update_tsdf::<StdOps>(-0.5, 1, 0.5, 10).0.to_bits(), 0);
    // sdf = d - z is +0 when d == z, never -0
    assert_eq!((1.25f32 - 1.25).to_bits(), 0);
    // end to end: a voxel centre exactly on the measured depth stores +0 and
    // no voxel of the volume holds -0
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let mut vol = volume::<8>(2048);
    let g = [0, 0, 49];
    let z = vol.layout().centre_from_voxel(g)[2];
    let d = DepthImage::from_size_val(cam.sensor.size(), z, 1.0)?;
    let mut ws = TsdfWorkspace::default();
    vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    let v = vol.voxel(g).unwrap();
    assert_eq!((v.tsdf.to_bits(), v.weight), (0, 1));
    for k in vol.block_keys() {
        let b = vol.block(k).unwrap();
        assert!(b.tsdf.iter().all(|t| t.to_bits() != 0x8000_0000));
    }
    Ok(())
}

// ----------------------------------------------------------------- colour ----

fn pack_rgbw(rgb: [u8; 3], cw: u8) -> u32 {
    rgb[0] as u32 | (rgb[1] as u32) << 8 | (rgb[2] as u32) << 16 | (cw as u32) << 24
}

/// Section 8.3 with `libm`, round half up.
fn colour_spec(s: u32, rgb: [u8; 3], max_cw: u8) -> u32 {
    let cw0 = s >> 24;
    let rc = 1.0 / ((cw0 + 1) as f32);
    let mut out = 0u32;
    for ch in 0..3 {
        let old = ((s >> (8 * ch)) & 0xff) as f32;
        let c = fma(rgb[ch] as f32 - old, rc, old);
        let hc = c + 0.5;
        let q = if hc >= 255.0 { 255 } else { hc as u32 };
        out |= q << (8 * ch);
    }
    out | ((cw0 + 1).min(max_cw as u32) << 24)
}

#[test]
fn tsdf_color_round_u8() {
    // the 255 boundary: 254.5 rounds up to 255, the clamp is explicit
    assert_eq!(tc::round_u8(254.5), 255);
    assert_eq!(tc::round_u8(254.49), 254);
    assert_eq!(tc::round_u8(255.0), 255);
    // c + 0.5 is one rounded add: 0.49999997 + 0.5 rounds up to 1.0
    assert_eq!(tc::round_u8(0.499_999_94), 0);
    assert_eq!(tc::round_u8(0.499_999_97), 1);
    assert_eq!(tc::round_u8(0.5), 1);
    // 10 -> 11 at weight 1: 10.5 rounds half up to 11 (half-even gives 10)
    let s = tc::update_color::<StdOps>(pack_rgbw([10, 0, 255], 1), [11, 0, 254], 16);
    assert_eq!(
        tc::unpack_rgb(s),
        [11, 0, 255],
        "254.5 + 0.5 is exactly 255"
    );
    assert_eq!(s >> 24, 2);
    // the colour weight cap: no overflow of `cw << 24`
    let s = tc::update_color::<StdOps>(pack_rgbw([7, 8, 9], 255), [7, 8, 9], 255);
    assert_eq!((s >> 24, tc::unpack_rgb(s)), (255, [7, 8, 9]));
    let s = tc::update_color::<StdOps>(pack_rgbw([1, 2, 3], 16), [1, 2, 3], 16);
    assert_eq!(s >> 24, 16);
    // freeze at the cap 16 (rc = 1/17): a step of 8 is lost, a step of 10 is not
    let at_cap = pack_rgbw([100, 100, 100], 16);
    assert_eq!(
        tc::unpack_rgb(tc::update_color::<StdOps>(at_cap, [108, 100, 100], 16))[0],
        100
    );
    assert_eq!(
        tc::unpack_rgb(tc::update_color::<StdOps>(at_cap, [110, 100, 100], 16))[0],
        101
    );
    assert_eq!(
        tc::unpack_rgb(tc::update_color::<StdOps>(at_cap, [92, 100, 100], 16))[0],
        100
    );
    assert_eq!(
        tc::unpack_rgb(tc::update_color::<StdOps>(at_cap, [90, 100, 100], 16))[0],
        99
    );
    // random states against the spec, with the wrong variants as witnesses
    let mut s = 12u64;
    let (mut mean, mut even, mut trunc) = (0, 0, 0);
    for _ in 0..100_000 {
        let v = rng(&mut s);
        let max_cw = [1u8, 16, 255][(v % 3) as usize];
        let cw = ((v >> 8) as u32 % (max_cw as u32 + 1)) as u8;
        let st = pack_rgbw([(v >> 16) as u8, (v >> 24) as u8, (v >> 32) as u8], cw);
        let obs = [(v >> 40) as u8, (v >> 48) as u8, (v >> 56) as u8];
        let got = tc::update_color::<StdOps>(st, obs, max_cw);
        assert_eq!(got, colour_spec(st, obs, max_cw));
        let old = tc::unpack_rgb(st);
        for ch in 0..3 {
            let c = fma(
                obs[ch] as f32 - old[ch] as f32,
                1.0 / (cw as u32 + 1) as f32,
                old[ch] as f32,
            );
            let q = tc::unpack_rgb(got)[ch];
            // the count that drives the mean is uncapped: cw + 1, also at the cap
            let capped = fma(
                obs[ch] as f32 - old[ch] as f32,
                1.0 / ((cw as u32 + 1).min(max_cw as u32)) as f32,
                old[ch] as f32,
            );
            mean += ((capped + 0.5) as u32 != q as u32) as usize;
            even += (c.round_ties_even().min(255.0) as u32 != q as u32) as usize;
            trunc += (c as u32 != q as u32) as usize;
        }
    }
    witness("capped colour count", mean);
    witness("round half to even", even);
    witness("truncation", trunc);
}

// -------------------------------------------------------- single rounding ----

#[test]
fn tsdf_params_single_rounding() -> R {
    let mut s = 1u64;
    let (mut inv_f32, mut az_f32, mut off_f32, mut t_f32) = (0, 0, 0, 0);
    for i in 0..4000 {
        // pinhole: (1 / fx) in f64, rounded once
        let fx = 100.0 + 900.0 * (rng(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let camera = PinholeCamera {
            fx,
            fy: fx + 1.0,
            cx: 10.5,
            cy: 7.5,
            ..PinholeCamera::IDENTITY
        };
        let size = ImageSize {
            width: 21,
            height: 15,
        };
        let sensor = PinholeSensor::new(&camera, size, DepthRange::new(0.1, 9.0)?)?;
        let p = sensor.params(ID, ID, 1.0);
        assert_eq!(p.fx.to_bits(), (fx as f32).to_bits());
        assert_eq!(p.inv_fx.to_bits(), ((1.0 / fx) as f32).to_bits());
        assert_eq!(p.inv_fy.to_bits(), ((1.0 / (fx + 1.0)) as f32).to_bits());
        inv_f32 += (p.inv_fx.to_bits() != (1.0f32 / (fx as f32)).to_bits()) as usize;

        // LiDAR: inv_az_step = (1 / step) and az_offset = (-start / step), in f64
        let step = (0.001 + 0.01 * (rng(&mut s) >> 11) as f64 / (1u64 << 53) as f64)
            * if i % 2 == 0 { 1.0 } else { -1.0 };
        let start = -3.0 + 6.0 * (rng(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let l = SphericalLidar::uniform(
            ImageSize {
                width: 64,
                height: 4,
            },
            AzimuthGrid {
                start,
                step,
                wrap: false,
            },
            0.2,
            -0.2,
            DepthRange::new(0.1, 9.0)?,
        )?;
        let t = l.tables();
        assert_eq!(t.inv_az_step.to_bits(), ((1.0 / step) as f32).to_bits());
        assert_eq!(t.az_offset.to_bits(), ((-start / step) as f32).to_bits());
        az_f32 += (t.inv_az_step.to_bits() != (1.0f32 / (step as f32)).to_bits()) as usize;
        off_f32 += (t.az_offset.to_bits() != (-(start as f32) / (step as f32)).to_bits()) as usize;

        // pose: sensor_from_volume is the f64 inverse rounded once, not -R^T t in f32
        let pz = pose(
            0.9 * (i as f64 * 0.01).sin(),
            0.4,
            0.2 * (i as f64 * 0.003).cos(),
            [
                10.0 * (i as f64 * 0.7).sin(),
                3.3 * (i as f64 * 0.2).cos(),
                1.0 + 0.001 * i as f64,
            ],
        );
        let depth = DepthImage::from_size_val(size, 1.0f32, 1.0)?;
        let rs = RangeSensor::from(sensor.clone());
        let o = prepare(&rs, &pz, (&depth).into(), None)?;
        let ObsKind::Pinhole(pp) = o.kind else {
            unreachable!()
        };
        let inv = pz.inverse();
        let want_t = [
            inv.translation.x as f32,
            inv.translation.y as f32,
            inv.translation.z as f32,
        ];
        assert_eq!(bits3(pp.sensor_from_volume.t), bits3(want_t));
        // R[r][c] is element r of the image of the c-th basis vector
        let rf = |r: usize, c: usize| -> f32 {
            let e = [
                Vec3F64::new(1.0, 0.0, 0.0),
                Vec3F64::new(0.0, 1.0, 0.0),
                Vec3F64::new(0.0, 0.0, 1.0),
            ][c];
            let v = pz.rotation * e;
            [v.x, v.y, v.z][r] as f32
        };
        let tf = [
            pz.translation.x as f32,
            pz.translation.y as f32,
            pz.translation.z as f32,
        ];
        let in_f32: [f32; 3] =
            std::array::from_fn(|r| -(rf(0, r) * tf[0] + rf(1, r) * tf[1] + rf(2, r) * tf[2]));
        t_f32 += (bits3(in_f32) != bits3(pp.sensor_from_volume.t)) as usize;
        // rows: r[3 i + j] = R[i][j] (row-major), and the inverse is the exact transpose
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    pp.volume_from_sensor.r[3 * r + c].to_bits(),
                    rf(r, c).to_bits()
                );
                assert_eq!(
                    pp.sensor_from_volume.r[3 * c + r].to_bits(),
                    pp.volume_from_sensor.r[3 * r + c].to_bits()
                );
            }
        }
    }
    witness("inv_fx from f32", inv_f32);
    witness("inv_az_step from f32", az_f32);
    witness("az_offset from f32", off_f32);
    witness("sensor_from_volume.t in f32", t_f32);
    // f32-sourced constants are derived in f32
    for vs in [0.01f32, 0.02, 0.025, 0.0123, 0.3] {
        let l = TsdfLayout::new(vs, 4.0)?;
        let g = l.grid(3);
        assert_eq!(g.inv_voxel.to_bits(), (1.0f32 / vs).to_bits());
        assert_eq!(g.half_voxel.to_bits(), (vs * 0.5).to_bits());
        assert_eq!(g.trunc.to_bits(), (4.0f32 * vs).to_bits());
        assert_eq!(g.inv_trunc.to_bits(), (1.0f32 / (4.0f32 * vs)).to_bits());
    }
    Ok(())
}
