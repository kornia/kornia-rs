//! The parallel touch pass (`TsdfKernel::Simd`) against the serial one: after
//! every step the hit list, the new-pool list (both in order), the miss set
//! and the counters match. Images span several chunks with a partial last one.

use super::*;

fn sorted(v: &[u64]) -> Vec<u64> {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

fn touch_same<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let cam = pinhole(241, 173, 200.0, (0.1, 4.0));
    let small = pinhole(96, 72, 90.0, (0.1, 4.0));
    let lid = lidar(1024, 40, 0.3, -0.3, (0.2, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.0, 0.1, 1.0],
            radius: 0.2,
        },
    ];
    let mut data = Vec::new();
    for k in 0..4 {
        let f = k as f64;
        let body = pose(0.0, 0.05 * f, 0.0, [0.04 * f, 0.0, 0.02 * f]);
        let ps = body.compose(&pose(0.0, 0.15, 0.0, [-0.1, 0.0, 0.05]));
        let pl = body.compose(&lidar_mount([0.0, 0.05, 0.0]));
        data.push((
            [body, ps, pl],
            render(&cam, &body, &scene),
            render(&small, &ps, &scene),
            render(&lid, &pl, &scene),
        ));
    }
    let (mut a, mut b) = (volume::<B>(16384), volume::<B>(16384));
    let mut wa = TsdfWorkspace::new(TsdfKernel::Scalar);
    let mut wb = TsdfWorkspace::new(TsdfKernel::Simd);
    let mut new_total = 0;
    for (k, (p, dc, ds, dl)) in data.iter().enumerate() {
        let step = [
            obs(&cam, p[0], dc),
            obs(&lid, p[2], dl),
            obs(&small, p[1], ds),
        ];
        let sa = a.integrate(&mut wa, &step)?;
        let sb = b.integrate(&mut wb, &step)?;
        assert_eq!(sa, sb, "B{B} step {k}: counters");
        assert_eq!(wa.visible, wb.visible, "B{B} step {k}: hit order");
        assert_eq!(wa.visible_new, wb.visible_new, "B{B} step {k}: new order");
        assert_eq!(
            sorted(&wa.obs_misses),
            sorted(&wb.obs_misses),
            "B{B} step {k}: misses"
        );
        for (ra, rb) in wa.ranges.iter().zip(&wb.ranges) {
            assert_eq!(
                (ra.visible, ra.new),
                (rb.visible, rb.new),
                "B{B} step {k}: ranges"
            );
        }
        new_total += sa.new_blocks;
        if k > 0 {
            assert!(!wa.visible.is_empty(), "B{B} step {k}: no warm hits");
        }
    }
    assert!(new_total > 100, "B{B}: {new_total} new blocks");
    assert!(a.block_keys().eq(b.block_keys()), "B{B}: block keys");
    Ok(())
}

// the parallel pass runs only inside a pool of >1 thread
fn in_pool(f: fn() -> R) -> R {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| f().map_err(|e| e.to_string()))?;
    Ok(())
}

#[test]
fn tsdf_touch_par_matches_serial_b8() -> R {
    in_pool(touch_same::<8>)
}

#[test]
fn tsdf_touch_par_matches_serial_b16() -> R {
    in_pool(touch_same::<16>)
}
