//! Pins of the multi-agent and submap rules (contract sections 3.5, 5.8, 10,
//! 11): spawn criteria, fold order, error atomicity, pool order per step,
//! residency, registration and validation.

use super::*;

const PI: f64 = std::f64::consts::PI;

fn at_x(x: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, 0.0, 0.0))
}

fn yaw(a: f64) -> Pose3d {
    Pose3d::new(mat(0.0, 0.0, a), Vec3F64::ZERO)
}

/// One 32 x 24 camera as sensor 0 of every agent.
fn cam_rig() -> SensorRig {
    SensorRig::new(vec![RigSensor {
        id: SensorId(0),
        sensor: pinhole(32, 24, 30.0, (0.1, 4.0)).sensor,
        body_from_sensor: Pose3d::IDENTITY,
    }])
    .unwrap()
}

/// A plane at 1 m over the whole image (about 35 blocks) ...
fn full_frame() -> DepthImage<f32> {
    DepthImage::from_size_val(
        ImageSize {
            width: 32,
            height: 24,
        },
        1.0f32,
        1.0,
    )
    .unwrap()
}

/// ... and a single valid pixel (at most 8 blocks).
fn one_pixel() -> DepthImage<f32> {
    let mut d = DepthImage::from_size_val(
        ImageSize {
            width: 32,
            height: 24,
        },
        0.0f32,
        1.0,
    )
    .unwrap();
    d.image_mut().as_slice_mut()[12 * 32 + 16] = 1.0;
    d
}

fn sparams(
    max_submaps: usize,
    growth: PoolGrowth,
    cap: usize,
    budget: Option<usize>,
) -> SubmapParams {
    SubmapParams {
        layout: TsdfLayout::new(VOXEL, 4.0).unwrap(),
        volume: TsdfParams {
            color: false,
            growth,
            ..params(cap)
        },
        arena_blocks: 4096,
        max_submaps,
        max_resident_bytes: budget,
    }
}

fn mini(params: SubmapParams, agents: &[(u32, SpawnPolicy)]) -> SubmapTsdf<8> {
    let mut map = SubmapTsdf::<8>::new(params).unwrap();
    for &(a, spawn) in agents {
        map.add_agent(AgentId(a), cam_rig(), spawn).unwrap();
    }
    map
}

fn frame(sensor: u32, d: &DepthImage<f32>) -> SensorFrame<'_> {
    SensorFrame {
        sensor: SensorId(sensor),
        depth: d.into(),
        color: None,
    }
}

fn run(
    map: &mut SubmapTsdf<8>,
    ws: &mut TsdfWorkspace,
    agent: u32,
    stamp: u64,
    body: Pose3d,
    d: &DepthImage<f32>,
) -> Result<StepReport, TsdfError> {
    let f = [frame(0, d)];
    map.integrate(
        ws,
        &AgentStep {
            agent: AgentId(agent),
            stamp,
            world_from_body: body,
            frames: &f,
        },
    )
}

fn id(agent: u32, seq: u32) -> SubmapId {
    SubmapId {
        agent: AgentId(agent),
        seq,
    }
}

fn hashes(map: &SubmapTsdf<8>) -> Vec<(SubmapId, u64, Pose3d)> {
    snapshot(map)
}

// ---------------------------------------------------------------- spawn ----

fn spawn_flags(policy: SpawnPolicy, bodies: &[Pose3d]) -> Vec<bool> {
    let d = one_pixel();
    let mut map = mini(sparams(32, PoolGrowth::Geometric, 64, None), &[(0, policy)]);
    let mut ws = TsdfWorkspace::default();
    bodies
        .iter()
        .enumerate()
        .map(|(i, b)| {
            run(&mut map, &mut ws, 0, i as u64 + 1, *b, &d)
                .unwrap()
                .spawned
        })
        .collect()
}

#[test]
fn tsdf_submap_spawn_boundaries() -> R {
    let policy = SpawnPolicy::default();
    // the first step of an agent always spawns; nothing else does by default
    assert_eq!(
        spawn_flags(policy, &[at_x(0.0), at_x(50.0), yaw(3.0)]),
        [true, false, false]
    );
    // translation is measured from the spawn pose, not the previous step
    let t = SpawnPolicy {
        max_translation: 0.5,
        ..policy
    };
    assert_eq!(
        spawn_flags(t, &[at_x(0.0), at_x(0.3), at_x(0.6), at_x(0.9)]),
        [true, false, true, false]
    );
    // |t|^2 >= max^2: exactly on the limit spawns, a hair below does not
    assert_eq!(spawn_flags(t, &[at_x(0.0), at_x(0.5)]), [true, true]);
    assert_eq!(
        spawn_flags(t, &[at_x(0.0), at_x(0.499_999_9)]),
        [true, false]
    );
    // rotation: (tr(R_s^T R) - 1) / 2 <= cos(max)
    let r = SpawnPolicy {
        max_rotation: 0.4,
        ..policy
    };
    assert_eq!(
        spawn_flags(r, &[yaw(0.0), yaw(0.3), yaw(0.5)]),
        [true, false, true]
    );
    let r0 = SpawnPolicy {
        max_rotation: 0.0,
        ..policy
    };
    assert_eq!(
        spawn_flags(r0, &[yaw(0.0), yaw(0.0), yaw(0.0)]),
        [true, true, true]
    );
    let rpi = SpawnPolicy {
        max_rotation: PI,
        ..policy
    };
    assert_eq!(spawn_flags(rpi, &[yaw(0.0), yaw(3.0)]), [true, false]);
    // blocks: >= max_blocks of the active submap
    let b1 = SpawnPolicy {
        max_blocks: 1,
        ..policy
    };
    assert_eq!(spawn_flags(b1, &[at_x(0.0); 3]), [true, true, true]);
    let b_big = SpawnPolicy {
        max_blocks: 100,
        ..policy
    };
    assert_eq!(spawn_flags(b_big, &[at_x(0.0); 3]), [true, false, false]);
    // steps: a submap takes max_steps steps, then the next one spawns
    let s3 = SpawnPolicy {
        max_steps: 3,
        ..policy
    };
    assert_eq!(
        spawn_flags(s3, &[at_x(0.0); 7]),
        [true, false, false, true, false, false, true]
    );
    // the step that spawns goes to the new submap only: ids count up per agent
    let d = one_pixel();
    let mut map = mini(sparams(8, PoolGrowth::Fixed, 64, None), &[(2, s3), (5, s3)]);
    let mut ws = TsdfWorkspace::default();
    let mut seqs = Vec::new();
    for k in 0..4u64 {
        // the other agent's steps never count toward this agent's max_steps
        seqs.push(run(&mut map, &mut ws, 2, 2 * k + 1, at_x(0.0), &d)?.submap);
        run(&mut map, &mut ws, 5, 2 * k + 2, at_x(0.0), &d)?;
    }
    assert_eq!(seqs, [id(2, 0), id(2, 0), id(2, 0), id(2, 1)]);
    assert_eq!(map.active_submap(AgentId(2)), Some(id(2, 1)));
    assert_eq!(map.active_submap(AgentId(9)), None);
    Ok(())
}

#[test]
fn tsdf_submap_frame_at_spawn() -> R {
    let d = one_pixel();
    let body = pose(0.0, 0.3, 0.0, [0.2, 0.0, 0.1]);
    let on_axis = BlockKey::new(0, 0, 6)?.bits();
    for (frame, in_submap_frame) in [
        (SubmapFrame::BodyAtSpawn, true),
        (SubmapFrame::TranslationAtSpawn, false),
    ] {
        let spawn = SpawnPolicy {
            frame,
            ..SpawnPolicy::default()
        };
        let mut map = mini(sparams(4, PoolGrowth::Fixed, 64, None), &[(0, spawn)]);
        let mut ws = TsdfWorkspace::default();
        let r = run(&mut map, &mut ws, 0, 1, body, &d)?;
        let p = map.submap_pose(r.submap).unwrap();
        if in_submap_frame {
            assert_eq!(p, body);
        } else {
            assert_eq!(p.rotation, Mat3F64::IDENTITY);
            assert_eq!(p.translation, body.translation);
        }
        // the pixel on the optical axis is 1 m ahead of the body in the
        // submap frame only when the submap frame is the body frame
        assert_eq!(
            map.pool_keys(r.submap).unwrap().contains(&on_axis),
            in_submap_frame
        );
        // later steps never move the submap
        run(&mut map, &mut ws, 0, 2, at_x(3.0), &d)?;
        assert_eq!(map.submap_pose(r.submap), Some(p));
    }
    Ok(())
}

// ---------------------------------------------------------- atomicity ----

#[test]
fn tsdf_submap_failed_step_changes_nothing() -> R {
    let (full, one) = (full_frame(), one_pixel());
    let mut map = mini(
        sparams(4, PoolGrowth::Fixed, 16, None),
        &[(0, SpawnPolicy::default())],
    );
    let mut ws = TsdfWorkspace::default();
    let e = run(&mut map, &mut ws, 0, 5, Pose3d::IDENTITY, &full).unwrap_err();
    assert!(matches!(e, TsdfError::PoolFull { capacity: 16, .. }), "{e}");
    assert_eq!(map.active_submap(AgentId(0)), None);
    assert_eq!(map.submap_ids().count(), 0);
    assert_eq!(map.memory_bytes(None).blocks, 0);
    // the stamp was not consumed and no submap number was burnt
    let r = run(&mut map, &mut ws, 0, 5, Pose3d::IDENTITY, &one)?;
    assert_eq!((r.submap, r.spawned), (id(0, 0), true));
    let before = hashes(&map);
    // an unknown sensor, a mirror pose, a wrong depth size: all rejected whole
    let bad_sensor = [frame(9, &one)];
    let unknown = AgentStep {
        agent: AgentId(0),
        stamp: 6,
        world_from_body: Pose3d::IDENTITY,
        frames: &bad_sensor,
    };
    let e = map.integrate(&mut ws, &unknown).unwrap_err();
    assert!(matches!(e, TsdfError::UnknownSensor(SensorId(9))));
    let e = run(&mut map, &mut ws, 0, 6, mirror_pose(), &one).unwrap_err();
    assert!(matches!(e, TsdfError::InvalidTransform));
    let small = DepthImage::from_size_val(
        ImageSize {
            width: 8,
            height: 6,
        },
        1.0f32,
        1.0,
    )?;
    let e = run(&mut map, &mut ws, 0, 6, Pose3d::IDENTITY, &small).unwrap_err();
    assert!(matches!(e, TsdfError::SizeMismatch { .. }));
    assert_eq!(hashes(&map), before);
    // stamp 6 is still free, and the step joins the active submap
    let r = run(&mut map, &mut ws, 0, 6, Pose3d::IDENTITY, &one)?;
    assert_eq!((r.submap, r.spawned), (id(0, 0), false));
    // a stale stamp is rejected (<=), and so is an unknown agent
    let e = run(&mut map, &mut ws, 0, 6, Pose3d::IDENTITY, &one).unwrap_err();
    assert!(matches!(
        e,
        TsdfError::StaleStamp {
            last: 6,
            got: 6,
            ..
        }
    ));
    let e = run(&mut map, &mut ws, 7, 1, Pose3d::IDENTITY, &one).unwrap_err();
    assert!(matches!(e, TsdfError::UnknownAgent(AgentId(7))));
    Ok(())
}

fn mirror_pose() -> Pose3d {
    Pose3d::new(
        Mat3F64::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Vec3F64::ZERO,
    )
}

#[test]
fn tsdf_submap_registration_and_validation() -> R {
    let d = one_pixel();
    let mut map = mini(
        sparams(4, PoolGrowth::Geometric, 64, None),
        &[(0, SpawnPolicy::default())],
    );
    let e = map
        .add_agent(AgentId(0), cam_rig(), SpawnPolicy::default())
        .unwrap_err();
    assert!(matches!(e, TsdfError::DuplicateAgent(AgentId(0))));
    // spawn policies: disabled criteria are the extremes, nonsense is rejected
    let base = SpawnPolicy::default();
    let bad = |p: SpawnPolicy, field: &str| {
        let mut m = SubmapTsdf::<8>::new(sparams(2, PoolGrowth::Fixed, 8, None)).unwrap();
        matches!(m.add_agent(AgentId(1), cam_rig(), p), Err(TsdfError::InvalidParameter { field: f, .. }) if f == field)
    };
    assert!(bad(
        SpawnPolicy {
            max_steps: 0,
            ..base
        },
        "max_steps"
    ));
    assert!(bad(
        SpawnPolicy {
            max_translation: f64::NAN,
            ..base
        },
        "max_translation"
    ));
    assert!(bad(
        SpawnPolicy {
            max_translation: -1.0,
            ..base
        },
        "max_translation"
    ));
    assert!(bad(
        SpawnPolicy {
            max_rotation: PI + 1e-9,
            ..base
        },
        "max_rotation"
    ));
    assert!(bad(
        SpawnPolicy {
            max_rotation: -0.1,
            ..base
        },
        "max_rotation"
    ));
    assert!(bad(
        SpawnPolicy {
            max_rotation: f64::NAN,
            ..base
        },
        "max_rotation"
    ));
    assert!(bad(
        SpawnPolicy {
            max_blocks: 0,
            ..base
        },
        "max_blocks"
    ));
    // arena and submap budgets
    let p = |f: &dyn Fn(&mut SubmapParams)| {
        let mut q = sparams(2, PoolGrowth::Fixed, 8, None);
        f(&mut q);
        SubmapTsdf::<8>::new(q).err()
    };
    let field = |e: Option<TsdfError>, name: &str| matches!(e, Some(TsdfError::InvalidParameter { field, .. }) if field == name);
    assert!(field(p(&|q| q.arena_blocks = 0), "arena_blocks"));
    assert!(field(p(&|q| q.max_submaps = 0), "max_submaps"));
    assert!(field(
        p(&|q| q.max_resident_bytes = Some(0)),
        "max_resident_bytes"
    ));
    // filters name agents and submaps that exist
    let mut ws = TsdfWorkspace::default();
    run(&mut map, &mut ws, 0, 1, Pose3d::IDENTITY, &d)?;
    let mut mesh = TriangleMesh::new();
    let mp = MeshParams::default();
    let e = map
        .extract_mesh_into(&mut ws, SubmapFilter::Agents(&[AgentId(3)]), &mp, &mut mesh)
        .unwrap_err();
    assert!(matches!(e, TsdfError::UnknownAgent(AgentId(3))));
    let e = map
        .extract_mesh_into(&mut ws, SubmapFilter::Submaps(&[id(0, 4)]), &mp, &mut mesh)
        .unwrap_err();
    assert!(matches!(e, TsdfError::UnknownSubmap(s) if s == id(0, 4)));
    // mesh parameters: weight_threshold >= 1, colours need colour storage
    let e = map
        .extract_mesh_into(
            &mut ws,
            SubmapFilter::All,
            &MeshParams {
                weight_threshold: 0,
                ..mp
            },
            &mut mesh,
        )
        .unwrap_err();
    assert!(matches!(
        e,
        TsdfError::InvalidParameter {
            field: "weight_threshold",
            ..
        }
    ));
    let e = map
        .extract_mesh_into(
            &mut ws,
            SubmapFilter::All,
            &MeshParams { colors: true, ..mp },
            &mut mesh,
        )
        .unwrap_err();
    assert!(matches!(
        e,
        TsdfError::InvalidParameter {
            field: "colors",
            ..
        }
    ));
    // loop closure poses: one bad pose rejects the whole update
    let sid = id(0, 0);
    let before = map.submap_pose(sid).unwrap();
    let moved = pose(0.0, 0.0, 0.2, [1.0, 2.0, 3.0]);
    for (bad_id, bad_pose, want) in [
        (sid, mirror_pose(), "invalid"),
        (id(0, 9), Pose3d::IDENTITY, "unknown"),
    ] {
        let r = map.set_submap_poses(&[
            SubmapPose {
                id: sid,
                world_from_submap: moved,
            },
            SubmapPose {
                id: bad_id,
                world_from_submap: bad_pose,
            },
        ]);
        match want {
            "invalid" => assert!(matches!(r, Err(TsdfError::InvalidTransform))),
            _ => assert!(matches!(r, Err(TsdfError::UnknownSubmap(_)))),
        }
        assert_eq!(map.submap_pose(sid), Some(before), "{want}");
    }
    map.set_submap_poses(&[SubmapPose {
        id: sid,
        world_from_submap: moved,
    }])?;
    assert_eq!(map.submap_pose(sid), Some(moved));
    Ok(())
}

// --------------------------------------------------------- batch order ----

#[test]
fn tsdf_batch_order_is_stamp_then_agent() -> R {
    let d = one_pixel();
    let fr = [frame(0, &d)];
    let st = |agent: u32, stamp: u64| AgentStep {
        agent: AgentId(agent),
        stamp,
        world_from_body: Pose3d::IDENTITY,
        frames: &fr,
    };
    let two = || {
        mini(
            sparams(1, PoolGrowth::Fixed, 64, None),
            &[(0, SpawnPolicy::default()), (1, SpawnPolicy::default())],
        )
    };
    let mut ws = TsdfWorkspace::default();
    // only one submap fits: whichever step sorts first is applied, the other fails.
    // Input order is the opposite of the sorted order in both cases.
    // (stamp, agent): agent 1 at stamp 10 goes before agent 0 at stamp 20
    // (an (agent, stamp) order would apply agent 0)
    let mut m = two();
    let e = m
        .integrate_batch(&mut ws, &[st(0, 20), st(1, 10)])
        .unwrap_err();
    assert!(
        matches!(&e, TsdfError::Batch { index: 0, source } if matches!(**source, TsdfError::TooManySubmaps { max: 1 })),
        "{e}"
    );
    assert_eq!(m.submap_ids().collect::<Vec<_>>(), [id(1, 0)]);
    // a shared stamp is ordered by agent: agent 0 goes first
    let mut m = two();
    let e = m
        .integrate_batch(&mut ws, &[st(1, 10), st(0, 10)])
        .unwrap_err();
    assert!(matches!(&e, TsdfError::Batch { index: 0, .. }), "{e}");
    assert_eq!(m.submap_ids().collect::<Vec<_>>(), [id(0, 0)]);
    // a batch that validates applies every step and reports the sums
    let mut m = mini(
        sparams(4, PoolGrowth::Fixed, 64, None),
        &[(0, SpawnPolicy::default()), (1, SpawnPolicy::default())],
    );
    let r = m.integrate_batch(&mut ws, &[st(1, 3), st(0, 2), st(0, 4)])?;
    assert_eq!((r.steps, r.spawned), (3, 2));
    assert!(r.stats.updated_voxels > 0 && r.stats.new_blocks > 0);
    // static validation rejects the whole batch before anything is applied
    let before = hashes(&m);
    let mirrored = AgentStep {
        world_from_body: mirror_pose(),
        ..st(0, 9)
    };
    let e = m
        .integrate_batch(&mut ws, &[st(1, 8), mirrored])
        .unwrap_err();
    assert!(
        matches!(&e, TsdfError::Batch { index: 1, source } if matches!(**source, TsdfError::InvalidTransform))
    );
    let e = m
        .integrate_batch(&mut ws, &[st(1, 8), st(0, 4)])
        .unwrap_err();
    assert!(
        matches!(&e, TsdfError::Batch { index: 1, source } if matches!(**source, TsdfError::StaleStamp { last: 4, got: 4, .. }))
    );
    let e = m
        .integrate_batch(&mut ws, &[st(1, 8), st(5, 9)])
        .unwrap_err();
    assert!(
        matches!(&e, TsdfError::Batch { index: 1, source } if matches!(**source, TsdfError::UnknownAgent(AgentId(5))))
    );
    assert_eq!(hashes(&m), before);
    Ok(())
}

#[test]
fn tsdf_batch_does_not_union_steps() -> R {
    // allocation is per step: a batch of two steps allocates like two calls,
    // so the second step's new blocks follow the first's even where their keys
    // sort lower
    let d = full_frame();
    let fr = [frame(0, &d)];
    let (p0, p1) = (at_x(0.0), at_x(-0.6));
    let st = |stamp: u64, body: Pose3d| AgentStep {
        agent: AgentId(0),
        stamp,
        world_from_body: body,
        frames: &fr,
    };
    let fresh = || {
        mini(
            sparams(4, PoolGrowth::Geometric, 256, None),
            &[(0, SpawnPolicy::default())],
        )
    };
    let mut ws = TsdfWorkspace::default();
    let mut batch = fresh();
    batch.integrate_batch(&mut ws, &[st(2, p1), st(1, p0)])?;
    let mut calls = fresh();
    calls.integrate(&mut ws, &st(1, p0))?;
    let n0 = calls.pool_keys(id(0, 0)).unwrap().len();
    calls.integrate(&mut ws, &st(2, p1))?;
    let (kb, kc) = (
        batch.pool_keys(id(0, 0)).unwrap(),
        calls.pool_keys(id(0, 0)).unwrap(),
    );
    assert_eq!(kb, kc, "pool order");
    assert_eq!(hashes(&batch), hashes(&calls));
    // each step's new keys ascend, and the second step's start below the first step's last
    let (first, second) = (&kc[..n0], &kc[n0..]);
    assert!(first.windows(2).all(|w| w[0] < w[1]) && second.windows(2).all(|w| w[0] < w[1]));
    assert!(
        second.len() > 4 && second[0] < first[first.len() - 1],
        "the steps must interleave in key order"
    );
    Ok(())
}

// ------------------------------------------------------------- residency ----

#[test]
fn tsdf_submap_residency_budget() -> R {
    let d = one_pixel();
    let spawn = SpawnPolicy {
        max_steps: 1,
        ..SpawnPolicy::default()
    };
    // every step spawns; last use order: (0,0) (1,0) (0,1) (1,1) (0,2)
    let order = [(0u32, 1u64), (1, 2), (0, 3), (1, 4), (0, 5)];
    let build = |budget: Option<usize>| -> Result<SubmapTsdf<8>, TsdfError> {
        let mut map = mini(
            sparams(8, PoolGrowth::Fixed, 64, budget),
            &[(0, spawn), (1, spawn)],
        );
        let mut ws = TsdfWorkspace::default();
        for (a, s) in order {
            run(&mut map, &mut ws, a, s, at_x(0.0), &d)?;
        }
        Ok(map)
    };
    let all = build(None)?;
    let bytes = |m: &SubmapTsdf<8>, s: SubmapId| m.submap_blocks(s).unwrap() * 512 * 6;
    let mut resident = Vec::new();
    all.resident_submaps_into(&mut resident);
    assert_eq!(resident, [id(0, 0), id(0, 1), id(0, 2), id(1, 0), id(1, 1)]);
    let active = [id(0, 2), id(1, 1)];
    let active_bytes: usize = active.iter().map(|&s| bytes(&all, s)).sum();
    // most recent non-active first: (0,1), then (1,0), then (0,0)
    let recent = [id(0, 1), id(1, 0), id(0, 0)];
    let before = hashes(&all);
    let expect = |k: usize| {
        let mut e: Vec<SubmapId> = active.iter().chain(&recent[..k]).copied().collect();
        e.sort_unstable();
        e
    };
    let mut budget = active_bytes;
    for k in 0..=recent.len() {
        let m = build(Some(budget))?;
        m.resident_submaps_into(&mut resident);
        assert_eq!(resident, expect(k), "budget {budget}");
        if k > 0 {
            // one byte short of admitting recent[k - 1]
            let m = build(Some(budget - 1))?;
            m.resident_submaps_into(&mut resident);
            assert_eq!(resident, expect(k - 1), "budget {} (one short)", budget - 1);
        }
        if k < recent.len() {
            budget += bytes(&all, recent[k]);
        }
    }
    // active submaps stay resident whatever the budget, and queries never alter content
    let tiny = build(Some(1))?;
    tiny.resident_submaps_into(&mut resident);
    assert_eq!(resident, active);
    assert_eq!(hashes(&all), before);
    assert_eq!(
        hashes(&tiny),
        before,
        "the budget only changes residency, not content"
    );
    Ok(())
}

// -------------------------------------------------------- growth policies ----

#[test]
fn tsdf_submap_growth_policies() -> R {
    let (full, one) = (full_frame(), one_pixel());
    let mut ws = TsdfWorkspace::default();
    // Fixed: a full arena fails the step and leaves the map untouched
    let mut p = sparams(4, PoolGrowth::Fixed, 4096, None);
    p.arena_blocks = 8;
    let mut map = mini(p, &[(0, SpawnPolicy::default())]);
    let e = run(&mut map, &mut ws, 0, 1, Pose3d::IDENTITY, &full).unwrap_err();
    assert!(matches!(e, TsdfError::PoolFull { capacity: 8, .. }), "{e}");
    assert_eq!(
        (map.submap_ids().count(), map.active_submap(AgentId(0))),
        (0, None)
    );
    // Geometric: the arena, the per-submap index and the header list all grow
    let spawn = SpawnPolicy {
        max_steps: 1,
        ..SpawnPolicy::default()
    };
    let mut p = sparams(1, PoolGrowth::Geometric, 4, None);
    p.arena_blocks = 8;
    let mut big = mini(sparams(8, PoolGrowth::Fixed, 4096, None), &[(0, spawn)]);
    let mut grown = mini(p, &[(0, spawn)]);
    for stamp in 1..=3 {
        let b = at_x(0.1 * stamp as f64);
        for m in [&mut big, &mut grown] {
            run(
                m,
                &mut ws,
                0,
                stamp,
                b,
                if stamp == 2 { &full } else { &one },
            )?;
        }
    }
    assert_eq!(grown.submap_ids().count(), 3);
    assert_eq!(
        hashes(&grown),
        hashes(&big),
        "growth never changes the stored bits"
    );
    assert!(grown.memory_bytes(None).blocks > 8);
    Ok(())
}

// ---------------------------------------------------- FP environment ----

/// Runs `f` on this thread with `bits` set in FPCR, then restores it.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn with_fpcr(bits: u64, f: impl FnOnce()) {
    let old: u64;
    // SAFETY: reads FPCR, then sets bits on this thread only; restored below.
    unsafe {
        std::arch::asm!("mrs {}, fpcr", out(reg) old);
        std::arch::asm!("msr fpcr, {}", in(reg) old | bits);
    }
    f();
    // SAFETY: restores this thread's original FPCR.
    unsafe {
        std::arch::asm!("msr fpcr, {}", in(reg) old);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn tsdf_fp_environment_rejected_by_every_entry() {
    use crate::registration::RgbdLevel;
    // flush-to-zero and a non-nearest rounding mode are both rejected, by
    // every public call, before anything is written
    for bits in [1u64 << 24, 1 << 22] {
        std::thread::spawn(move || {
            let (full, one) = (full_frame(), one_pixel());
            let cam = pinhole(32, 24, 30.0, (0.1, 4.0));
            let RangeSensor::Pinhole(ps) = &cam.sensor else {
                unreachable!()
            };
            let mut ws = TsdfWorkspace::default();
            let mut vol = volume::<8>(256);
            vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &full)])
                .unwrap();
            let mut map = mini(
                sparams(4, PoolGrowth::Geometric, 64, None),
                &[(0, SpawnPolicy::default())],
            );
            run(&mut map, &mut ws, 0, 1, Pose3d::IDENTITY, &one).unwrap();
            let (hv, hm) = (vol.content_hash(), hashes(&map));
            let lid = lidar(64, 8, 0.2, -0.2, (0.2, 5.0));
            let RangeSensor::Spherical(sl) = &lid.sensor else {
                unreachable!()
            };
            let rp = RaycastParams::new(vol.layout(), 0.1, 4.0).unwrap();
            let mp = MeshParams::default();
            let sz = cam.sensor.size();
            let (mut maps, mut mesh, mut pts) = (
                RaycastMaps::new(sz, false).unwrap(),
                TriangleMesh::new(),
                Vec::new(),
            );
            let mut level = RgbdLevel::zeroed(ps.camera(), sz).unwrap();
            let mut pyr = RgbdPyramid::zeroed(ps.camera(), sz, 2).unwrap();
            let (mut cache, mut scache) = (BlockMeshCache::new(), SubmapMeshCache::new());
            let mut range = DepthImage::from_size_val(sl.size(), 0.0f32, 1.0).unwrap();
            let f = [frame(0, &one)];
            let st = AgentStep {
                agent: AgentId(0),
                stamp: 2,
                world_from_body: Pose3d::IDENTITY,
                frames: &f,
            };
            let w = Pose3d::IDENTITY;
            let all = SubmapFilter::All;
            with_fpcr(bits, || {
                let fp = |r: Result<(), TsdfError>| matches!(r, Err(TsdfError::FpEnvironment));
                assert!(fp(vol
                    .integrate(&mut ws, &[obs(&cam, w, &full)])
                    .map(|_| ())));
                assert!(fp(vol.raycast_into(
                    &mut ws,
                    ps.camera(),
                    &w,
                    &rp,
                    &mut maps
                )));
                assert!(fp(vol.raycast_level_into(&mut ws, &w, &rp, &mut level)));
                assert!(fp(vol.raycast_pyramid_into(&mut ws, &w, &rp, &mut pyr)));
                assert!(fp(vol.extract_mesh_into(&mut ws, &mp, &mut mesh)));
                assert!(fp(vol
                    .update_mesh_cache(&mut ws, &mp, &mut cache)
                    .map(|_| ())));
                assert!(fp(
                    vol.extract_point_cloud_into(&mut ws, &mp, &mut pts, None, None)
                ));
                assert!(fp(
                    sl.range_image_from_points_into(&[[3.0, 0.0, 0.0]], &mut range)
                ));
                assert!(fp(map.integrate(&mut ws, &st).map(|_| ())));
                assert!(fp(map.integrate_batch(&mut ws, &[st]).map(|_| ())));
                assert!(fp(map.extract_mesh_into(&mut ws, all, &mp, &mut mesh)));
                assert!(fp(map
                    .update_mesh_cache(&mut ws, &mp, &mut scache)
                    .map(|_| ())));
                assert!(fp(scache.write_into(&map, all, &mut mesh)));
                assert!(fp(map.raycast_into(
                    &mut ws,
                    ps.camera(),
                    &w,
                    all,
                    &rp,
                    &mut maps
                )));
                assert!(fp(map.raycast_pyramid_into(&mut ws, &w, all, &rp, &mut pyr)));
            });
            assert_eq!((vol.content_hash(), hashes(&map)), (hv, hm));
            // the same calls work in the default environment
            vol.raycast_into(&mut ws, ps.camera(), &w, &rp, &mut maps)
                .unwrap();
            map.integrate(&mut ws, &st).unwrap();
        })
        .join()
        .unwrap();
    }
}
