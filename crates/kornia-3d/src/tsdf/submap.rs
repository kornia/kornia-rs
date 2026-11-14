//! Submaps: one active TSDF submap per agent, spawned by a per-agent
//! policy, moved rigidly by loop closure, queried as a union.
//!
//! Every submap has its own block index; voxel pages come from one shared
//! arena. Each agent alone writes its submaps, and a batch is applied in
//! `(stamp, agent)` order (sensors of a step in ascending id), so the
//! submap bytes do not depend on how the agents' steps interleave.

use std::marker::PhantomData;

use kornia_image::Image;

use super::alloc::prepare;
use super::integrate::{integrate_step, StepTarget};
use super::mesh::{extract_view, update_cache, BlockMeshCache, MeshParams, MeshUpdateStats};
use super::raycast::{
    check_raycast, core_params, level_out, maps_out, raycast_views, RayOut, RaycastMaps,
    RaycastParams,
};
use super::sensor::{check_observation, DepthRef, RangeSensor};
use super::volume::{content_hash, BlockIndex, Pools, VolumeRef};
use super::workspace::TsdfWorkspace;
use super::{
    check_fp_env, check_pose, max_blocks, rigid_from_pose, AgentId, BlockDim, IntegrateStats,
    PoolGrowth, SensorId, SubmapId, SupportedBlockDim, TsdfError, TsdfLayout, TsdfParams,
};
use crate::camera::PinholeCamera;
use crate::mesh::TriangleMesh;
use crate::pose::Pose3d;
use crate::registration::RgbdPyramid;
use kornia_image::ImageSize;

/// One sensor of a rig.
#[derive(Debug, Clone)]
pub struct RigSensor {
    /// Id, unique within the rig.
    pub id: SensorId,
    /// The sensor.
    pub sensor: RangeSensor,
    /// Sensor pose in the agent's body frame.
    pub body_from_sensor: Pose3d,
}

/// An agent's sensors, sorted by id.
#[derive(Debug, Clone)]
pub struct SensorRig {
    sensors: Vec<RigSensor>,
}

impl SensorRig {
    /// Validates a rig.
    ///
    /// # Arguments
    ///
    /// * `sensors` - The sensors, any order.
    ///
    /// # Returns
    ///
    /// The rig, sensors sorted by id.
    ///
    /// # Errors
    ///
    /// [`TsdfError::DuplicateSensor`] or [`TsdfError::InvalidTransform`].
    pub fn new(mut sensors: Vec<RigSensor>) -> Result<Self, TsdfError> {
        sensors.sort_unstable_by_key(|s| s.id);
        if let Some(w) = sensors.windows(2).find(|w| w[0].id == w[1].id) {
            return Err(TsdfError::DuplicateSensor(w[0].id));
        }
        for s in &sensors {
            check_pose(&s.body_from_sensor)?;
        }
        Ok(Self { sensors })
    }

    /// The sensors, ascending id.
    pub fn sensors(&self) -> &[RigSensor] {
        &self.sensors
    }

    /// A sensor by id.
    ///
    /// # Arguments
    ///
    /// * `id` - Sensor id.
    ///
    /// # Returns
    ///
    /// The sensor, `None` if it is not in the rig.
    pub fn get(&self, id: SensorId) -> Option<&RigSensor> {
        self.sensors
            .binary_search_by_key(&id, |s| s.id)
            .ok()
            .map(|i| &self.sensors[i])
    }
}

/// One sensor's image in an [`AgentStep`].
#[derive(Clone, Copy)]
pub struct SensorFrame<'a> {
    /// Sensor of the agent's rig.
    pub sensor: SensorId,
    /// Depth (pinhole) or range (spherical) image of the sensor's size.
    pub depth: DepthRef<'a>,
    /// Registered colour (pinhole only).
    pub color: Option<&'a Image<u8, 3>>,
}

/// One agent's measurements at one stamp.
#[derive(Clone, Copy)]
pub struct AgentStep<'a> {
    /// The agent.
    pub agent: AgentId,
    /// Stamp; strictly increasing per agent.
    pub stamp: u64,
    /// Body pose in the world frame.
    pub world_from_body: Pose3d,
    /// Frames, at most one per sensor, any order (applied by ascending id).
    pub frames: &'a [SensorFrame<'a>],
}

/// Frame of a newly spawned submap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubmapFrame {
    /// Identity rotation, origin at the body position.
    #[default]
    TranslationAtSpawn,
    /// The body pose of the spawning step.
    BodyAtSpawn,
}

/// When an agent starts a new submap (checked in `f64` before each step,
/// from the agent's own steps). `u32::MAX`, `f64::INFINITY` and
/// `usize::MAX` disable a criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpawnPolicy {
    /// Steps per submap, `>= 1`.
    pub max_steps: u32,
    /// Body travel since the spawn (metres), `>= 0`.
    pub max_translation: f64,
    /// Body rotation since the spawn (radians), in `[0, pi]`.
    pub max_rotation: f64,
    /// Blocks per submap, `>= 1`.
    pub max_blocks: usize,
    /// Frame of a new submap.
    pub frame: SubmapFrame,
}

impl Default for SpawnPolicy {
    /// Never spawn after the first submap.
    fn default() -> Self {
        Self {
            max_steps: u32::MAX,
            max_translation: f64::INFINITY,
            max_rotation: f64::INFINITY,
            max_blocks: usize::MAX,
            frame: SubmapFrame::default(),
        }
    }
}

/// [`SubmapTsdf`] parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmapParams {
    /// Grid geometry shared by every submap.
    pub layout: TsdfLayout,
    /// Voxel parameters; `block_capacity` is the per-submap block cap.
    pub volume: TsdfParams,
    /// Voxel pages of the shared arena.
    pub arena_blocks: usize,
    /// Submap slots (preallocated).
    pub max_submaps: usize,
    /// Byte budget of resident submaps for a device backend
    /// ([`SubmapTsdf::resident_submaps_into`]); `None` = unbounded.
    pub max_resident_bytes: Option<usize>,
}

/// A submap's pose for [`SubmapTsdf::set_submap_poses`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubmapPose {
    /// Submap.
    pub id: SubmapId,
    /// Its new pose in the world frame.
    pub world_from_submap: Pose3d,
}

/// Which submaps a query covers.
#[derive(Debug, Clone, Copy)]
pub enum SubmapFilter<'a> {
    /// Every submap.
    All,
    /// The submaps of these agents.
    Agents(&'a [AgentId]),
    /// These submaps.
    Submaps(&'a [SubmapId]),
}

impl SubmapFilter<'_> {
    fn matches(&self, id: SubmapId) -> bool {
        match self {
            Self::All => true,
            Self::Agents(a) => a.contains(&id.agent),
            Self::Submaps(s) => s.contains(&id),
        }
    }
}

/// Result of one [`SubmapTsdf::integrate`] step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepReport {
    /// Submap the step went into.
    pub submap: SubmapId,
    /// The step spawned that submap.
    pub spawned: bool,
    /// Integration counters.
    pub stats: IntegrateStats,
}

/// Result of [`SubmapTsdf::integrate_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchReport {
    /// Steps applied.
    pub steps: usize,
    /// Submaps spawned.
    pub spawned: usize,
    /// Summed integration counters.
    pub stats: IntegrateStats,
}

/// Memory held, see [`SubmapTsdf::memory_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryReport {
    /// Host bytes.
    pub host_bytes: usize,
    /// Device bytes (0 for the host type).
    pub device_bytes: usize,
    /// Allocated blocks.
    pub blocks: usize,
}

/// One mesh cache per submap, see [`SubmapTsdf::update_mesh_cache`].
#[derive(Debug, Default)]
pub struct SubmapMeshCache {
    caches: Vec<BlockMeshCache>,
    ids: Vec<Option<SubmapId>>,
}

impl SubmapMeshCache {
    /// An empty cache; each submap's slab grows on its first updates.
    ///
    /// # Returns
    ///
    /// The cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache with a slab reserved for each of `submaps` submap
    /// slots, so that updates within those capacities never allocate, also
    /// for a submap spawned later.
    ///
    /// # Arguments
    ///
    /// * `submaps` - Submap slots (the map's [`SubmapParams::max_submaps`]).
    /// * `vertices` - Vertex capacity of each slab.
    /// * `triangles` - Triangle capacity of each slab.
    /// * `blocks` - Block capacity of each slab.
    ///
    /// # Returns
    ///
    /// The cache.
    pub fn with_capacity(submaps: usize, vertices: usize, triangles: usize, blocks: usize) -> Self {
        Self {
            caches: (0..submaps)
                .map(|_| BlockMeshCache::with_capacity(vertices, triangles, blocks))
                .collect(),
            ids: vec![None; submaps],
        }
    }

    /// Writes the cached meshes of the selected submaps in the world frame
    /// (ascending submap id, `v_w = R v + t`, `n_w = R n`).
    ///
    /// # Arguments
    ///
    /// * `map` - The map the cache was updated from (for the poses).
    /// * `filter` - Submaps to write.
    /// * `mesh` - Output mesh (cleared first).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the meshes.
    ///
    /// # Errors
    ///
    /// [`TsdfError::UnknownAgent`], [`TsdfError::UnknownSubmap`],
    /// [`TsdfError::MeshTooLarge`], [`TsdfError::FpEnvironment`].
    pub fn write_into<const B: usize>(
        &self,
        map: &SubmapTsdf<B>,
        filter: SubmapFilter<'_>,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError>
    where
        BlockDim<B>: SupportedBlockDim,
    {
        check_fp_env()?;
        map.check_filter(&filter)?;
        mesh.clear();
        for &slot in &map.live {
            let h = &map.headers[slot as usize];
            if !filter.matches(h.id) {
                continue;
            }
            let s = slot as usize;
            if let (Some(c), Some(Some(id))) = (self.caches.get(s), self.ids.get(s)) {
                if *id == h.id {
                    let xf = rigid_from_pose(&h.world_from_submap)?;
                    c.append_to(mesh, Some(&xf))?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Header {
    id: SubmapId,
    index: BlockIndex,
    world_from_submap: Pose3d,
    steps: u32,
    last_used: u64,
}

impl Header {
    /// A free slot with a hash for `capacity` blocks.
    fn new(capacity: usize) -> Self {
        Self {
            id: SubmapId {
                agent: AgentId(0),
                seq: 0,
            },
            index: BlockIndex::with_capacity(capacity),
            world_from_submap: Pose3d::IDENTITY,
            steps: 0,
            last_used: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Agent {
    id: AgentId,
    rig: SensorRig,
    spawn: SpawnPolicy,
    cos_max_rotation: f64,
    active: Option<u32>,
    next_seq: u32,
    last_stamp: Option<u64>,
    spawn_body: Pose3d,
}

/// A multi-agent map of TSDF submaps over one shared voxel arena.
///
/// Each agent integrates into its own active submap; a [`SpawnPolicy`]
/// starts a new one. Loop closure moves submaps with
/// [`SubmapTsdf::set_submap_poses`] (poses from any optimiser, e.g.
/// [`crate::pgo`]); queries (mesh, raycast) cover a [`SubmapFilter`] of
/// the union, each submap in its own pose. Overlapping submaps are never
/// fused voxel-wise: the raycast keeps the nearest hit.
pub struct SubmapTsdf<const B: usize = 8>
where
    BlockDim<B>: SupportedBlockDim,
{
    params: SubmapParams,
    headers: Vec<Header>,
    used: usize,
    live: Vec<u32>,
    agents: Vec<Agent>,
    arena: Pools,
    clock: u64,
    _block: PhantomData<BlockDim<B>>,
}

impl<const B: usize> SubmapTsdf<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    const LOG2: u32 = <BlockDim<B> as SupportedBlockDim>::LOG2;
    const VOXELS: usize = <BlockDim<B> as SupportedBlockDim>::VOXELS;

    /// Creates an empty map and preallocates `max_submaps` submap slots
    /// (each with a hash for `volume.block_capacity` blocks) and the arena.
    ///
    /// # Arguments
    ///
    /// * `params` - Layout, voxel parameters and capacities.
    ///
    /// # Returns
    ///
    /// The map.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for a zero or oversized capacity.
    pub fn new(params: SubmapParams) -> Result<Self, TsdfError> {
        params.volume.validate(Self::VOXELS)?;
        let bad = |field: &'static str, value: f64| TsdfError::InvalidParameter { field, value };
        if params.arena_blocks == 0 || params.arena_blocks > max_blocks(Self::VOXELS) {
            return Err(bad("arena_blocks", params.arena_blocks as f64));
        }
        if params.max_submaps == 0 || params.max_submaps > u32::MAX as usize {
            return Err(bad("max_submaps", params.max_submaps as f64));
        }
        if params.max_resident_bytes == Some(0) {
            return Err(bad("max_resident_bytes", 0.0));
        }
        let cap = params.volume.block_capacity;
        Ok(Self {
            headers: (0..params.max_submaps).map(|_| Header::new(cap)).collect(),
            used: 0,
            live: Vec::with_capacity(params.max_submaps),
            agents: Vec::new(),
            arena: Pools::with_capacity(params.arena_blocks, Self::VOXELS, params.volume.color),
            clock: 0,
            params,
            _block: PhantomData,
        })
    }

    /// Map parameters.
    pub fn params(&self) -> &SubmapParams {
        &self.params
    }

    /// Registers an agent with its rig and spawn policy.
    ///
    /// # Arguments
    ///
    /// * `agent` - New agent id.
    /// * `rig` - Its sensors.
    /// * `spawn` - Its spawn policy.
    ///
    /// # Returns
    ///
    /// `Ok(())` once registered.
    ///
    /// # Errors
    ///
    /// [`TsdfError::DuplicateAgent`], [`TsdfError::InvalidParameter`]
    /// (policy).
    pub fn add_agent(
        &mut self,
        agent: AgentId,
        rig: SensorRig,
        spawn: SpawnPolicy,
    ) -> Result<(), TsdfError> {
        let at = match self.agents.binary_search_by_key(&agent, |a| a.id) {
            Ok(_) => return Err(TsdfError::DuplicateAgent(agent)),
            Err(i) => i,
        };
        let bad =
            |field: &'static str, value: f64| Err(TsdfError::InvalidParameter { field, value });
        if spawn.max_steps == 0 {
            return bad("max_steps", 0.0);
        }
        if spawn.max_translation.is_nan() || spawn.max_translation < 0.0 {
            return bad("max_translation", spawn.max_translation);
        }
        let rot = spawn.max_rotation;
        if !(rot == f64::INFINITY || (0.0..=std::f64::consts::PI).contains(&rot)) {
            return bad("max_rotation", rot);
        }
        if spawn.max_blocks == 0 {
            return bad("max_blocks", 0.0);
        }
        let cos_max_rotation = if rot == f64::INFINITY {
            f64::NEG_INFINITY
        } else {
            libm::cos(rot)
        };
        self.agents.insert(
            at,
            Agent {
                id: agent,
                rig,
                spawn,
                cos_max_rotation,
                active: None,
                next_seq: 0,
                last_stamp: None,
                spawn_body: Pose3d::IDENTITY,
            },
        );
        Ok(())
    }

    fn agent_index(&self, agent: AgentId) -> Result<usize, TsdfError> {
        self.agents
            .binary_search_by_key(&agent, |a| a.id)
            .map_err(|_| TsdfError::UnknownAgent(agent))
    }

    fn slot_of(&self, id: SubmapId) -> Option<usize> {
        self.live
            .binary_search_by_key(&id, |&s| self.headers[s as usize].id)
            .ok()
            .map(|i| self.live[i] as usize)
    }

    fn view(&self, slot: usize) -> VolumeRef<'_> {
        VolumeRef {
            index: &self.headers[slot].index,
            pools: &self.arena,
            grid: self.params.layout.grid(Self::LOG2),
        }
    }

    fn check_filter(&self, filter: &SubmapFilter<'_>) -> Result<(), TsdfError> {
        match filter {
            SubmapFilter::All => Ok(()),
            SubmapFilter::Agents(a) => a
                .iter()
                .try_for_each(|&id| self.agent_index(id).map(|_| ())),
            SubmapFilter::Submaps(s) => s.iter().try_for_each(|&id| {
                self.slot_of(id)
                    .map(|_| ())
                    .ok_or(TsdfError::UnknownSubmap(id))
            }),
        }
    }

    /// Static checks of a step (nothing mutated).
    fn validate_step(&self, step: &AgentStep<'_>) -> Result<(), TsdfError> {
        let a = &self.agents[self.agent_index(step.agent)?];
        if let Some(last) = a.last_stamp {
            if step.stamp <= last {
                return Err(TsdfError::StaleStamp {
                    agent: step.agent,
                    last,
                    got: step.stamp,
                });
            }
        }
        check_pose(&step.world_from_body)?;
        for (i, f) in step.frames.iter().enumerate() {
            let rs = a
                .rig
                .get(f.sensor)
                .ok_or(TsdfError::UnknownSensor(f.sensor))?;
            if step.frames[..i].iter().any(|g| g.sensor == f.sensor) {
                return Err(TsdfError::DuplicateSensor(f.sensor));
            }
            check_observation(&rs.sensor, &f.depth, f.color)?;
        }
        Ok(())
    }

    /// Spawn decision for `a` before `step`.
    fn should_spawn(&self, a: &Agent, step: &AgentStep<'_>) -> bool {
        let Some(slot) = a.active else {
            return true;
        };
        let h = &self.headers[slot as usize];
        let p = &a.spawn;
        let d = step.world_from_body.translation - a.spawn_body.translation;
        let rel = a.spawn_body.rotation.transpose() * step.world_from_body.rotation;
        let cos = (rel.x_axis.x + rel.y_axis.y + rel.z_axis.z - 1.0) / 2.0;
        h.steps >= p.max_steps
            || d.dot(d) >= p.max_translation * p.max_translation
            || cos <= a.cos_max_rotation
            || h.index.len() >= p.max_blocks
    }

    /// Applies one validated step.
    fn apply_step(
        &mut self,
        ws: &mut TsdfWorkspace,
        step: &AgentStep<'_>,
    ) -> Result<StepReport, TsdfError> {
        let ai = self.agent_index(step.agent)?;
        let spawn = self.should_spawn(&self.agents[ai], step);
        let slot = if spawn {
            if self.used == self.headers.len() {
                if self.params.volume.growth == PoolGrowth::Fixed {
                    return Err(TsdfError::TooManySubmaps {
                        max: self.headers.len(),
                    });
                }
                self.headers
                    .push(Header::new(self.params.volume.block_capacity));
            }
            self.headers[self.used].index.clear();
            self.used
        } else {
            self.agents[ai].active.unwrap_or_default() as usize
        };
        let world_from_submap = if spawn {
            match self.agents[ai].spawn.frame {
                SubmapFrame::TranslationAtSpawn => Pose3d::new(
                    kornia_algebra::Mat3F64::IDENTITY,
                    step.world_from_body.translation,
                ),
                SubmapFrame::BodyAtSpawn => step.world_from_body,
            }
        } else {
            self.headers[slot].world_from_submap
        };
        let submap_from_body = world_from_submap.inverse().compose(&step.world_from_body);

        let mut order = std::mem::take(&mut ws.frame_order);
        order.clear();
        order.extend(0..step.frames.len() as u32);
        order.sort_unstable_by_key(|&i| step.frames[i as usize].sensor);
        let rig = &self.agents[ai].rig;
        let obs = |s: usize| {
            let f = &step.frames[order[s] as usize];
            let rs = rig
                .get(f.sensor)
                .ok_or(TsdfError::UnknownSensor(f.sensor))?;
            let volume_from_sensor = submap_from_body.compose(&rs.body_from_sensor);
            prepare(&rs.sensor, &volume_from_sensor, f.depth, f.color)
        };
        let target = StepTarget {
            index: &mut self.headers[slot].index,
            pools: &mut self.arena,
            layout: &self.params.layout,
            params: &self.params.volume,
            log2: Self::LOG2,
        };
        let res = integrate_step(target, ws, step.frames.len(), &obs);
        ws.frame_order = order;
        let stats = res?;

        let a = &mut self.agents[ai];
        if spawn {
            let h = &mut self.headers[slot];
            h.id = SubmapId {
                agent: a.id,
                seq: a.next_seq,
            };
            h.world_from_submap = world_from_submap;
            h.steps = 0;
            a.next_seq += 1;
            a.active = Some(slot as u32);
            a.spawn_body = step.world_from_body;
            self.used += 1;
            let id = h.id;
            let headers = &self.headers;
            let at = self.live.partition_point(|&s| headers[s as usize].id < id);
            self.live.insert(at, slot as u32);
        }
        self.clock += 1;
        let h = &mut self.headers[slot];
        h.steps += 1;
        h.last_used = self.clock;
        a.last_stamp = Some(step.stamp);
        Ok(StepReport {
            submap: h.id,
            spawned: spawn,
            stats,
        })
    }

    /// Integrates one agent step into the agent's active submap, spawning
    /// a new submap first when the agent's policy says so.
    ///
    /// The step's frames are applied in ascending sensor id; each sensor's
    /// pose in the submap is `world_from_submap⁻¹ ∘ world_from_body ∘
    /// body_from_sensor` (in `f64`).
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `step` - The step.
    ///
    /// # Returns
    ///
    /// The submap used, whether it was spawned, and the counters.
    ///
    /// # Errors
    ///
    /// [`TsdfError::UnknownAgent`], [`TsdfError::StaleStamp`],
    /// [`TsdfError::UnknownSensor`], [`TsdfError::DuplicateSensor`],
    /// observation errors, [`TsdfError::PoolFull`],
    /// [`TsdfError::TooManySubmaps`]; on error the map is unchanged.
    pub fn integrate(
        &mut self,
        ws: &mut TsdfWorkspace,
        step: &AgentStep<'_>,
    ) -> Result<StepReport, TsdfError> {
        check_fp_env()?;
        self.validate_step(step)?;
        self.apply_step(ws, step)
    }

    /// Integrates several agents' steps in `(stamp, agent)` order, so any
    /// interleaving of the agents' streams gives the same submaps as a
    /// serial replay.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `steps` - Steps of any agents, any order.
    ///
    /// # Returns
    ///
    /// The number of steps and spawns, and the summed counters.
    ///
    /// # Errors
    ///
    /// [`TsdfError::Batch`] with the input index of the failing step. The
    /// whole batch is validated first (nothing applied on a validation
    /// error, including [`TsdfError::DuplicateStep`]); if a step fails
    /// while applying, the steps before it in `(stamp, agent)` order stay
    /// applied.
    pub fn integrate_batch(
        &mut self,
        ws: &mut TsdfWorkspace,
        steps: &[AgentStep<'_>],
    ) -> Result<BatchReport, TsdfError> {
        check_fp_env()?;
        let batch = |index: usize, e: TsdfError| TsdfError::Batch {
            index,
            source: Box::new(e),
        };
        for (i, s) in steps.iter().enumerate() {
            self.validate_step(s).map_err(|e| batch(i, e))?;
        }
        let mut order = std::mem::take(&mut ws.batch_order);
        order.clear();
        order.extend(0..steps.len() as u32);
        order.sort_unstable_by_key(|&i| (steps[i as usize].stamp, steps[i as usize].agent));
        let res = (|| {
            for w in order.windows(2) {
                let (a, b) = (&steps[w[0] as usize], &steps[w[1] as usize]);
                if (a.stamp, a.agent) == (b.stamp, b.agent) {
                    let e = TsdfError::DuplicateStep {
                        agent: b.agent,
                        stamp: b.stamp,
                    };
                    return Err(batch(w[1] as usize, e));
                }
            }
            let mut report = BatchReport {
                steps: 0,
                spawned: 0,
                stats: IntegrateStats::default(),
            };
            for &i in &order {
                let r = self
                    .apply_step(ws, &steps[i as usize])
                    .map_err(|e| batch(i as usize, e))?;
                report.steps += 1;
                report.spawned += r.spawned as usize;
                report.stats.accumulate(&r.stats);
            }
            Ok(report)
        })();
        ws.batch_order = order;
        res
    }

    /// Moves submaps (e.g. after a pose-graph optimisation); never
    /// re-integrates. All poses are validated before any is applied.
    ///
    /// # Arguments
    ///
    /// * `poses` - New world poses of some submaps.
    ///
    /// # Returns
    ///
    /// `Ok(())` once applied.
    ///
    /// # Errors
    ///
    /// [`TsdfError::UnknownSubmap`] or [`TsdfError::InvalidTransform`];
    /// nothing is applied then.
    pub fn set_submap_poses(&mut self, poses: &[SubmapPose]) -> Result<(), TsdfError> {
        for p in poses {
            self.slot_of(p.id).ok_or(TsdfError::UnknownSubmap(p.id))?;
            check_pose(&p.world_from_submap)?;
        }
        for p in poses {
            if let Some(s) = self.slot_of(p.id) {
                self.headers[s].world_from_submap = p.world_from_submap;
            }
        }
        Ok(())
    }

    /// Submap ids, ascending.
    pub fn submap_ids(&self) -> impl Iterator<Item = SubmapId> + '_ {
        self.live.iter().map(|&s| self.headers[s as usize].id)
    }

    /// The agent's active submap.
    ///
    /// # Arguments
    ///
    /// * `agent` - Agent id.
    ///
    /// # Returns
    ///
    /// Its active submap, `None` before its first step or if unknown.
    pub fn active_submap(&self, agent: AgentId) -> Option<SubmapId> {
        let a = &self.agents[self.agent_index(agent).ok()?];
        a.active.map(|s| self.headers[s as usize].id)
    }

    /// A submap's world pose.
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// `world_from_submap`, `None` for an unknown id.
    pub fn submap_pose(&self, id: SubmapId) -> Option<Pose3d> {
        self.slot_of(id).map(|s| self.headers[s].world_from_submap)
    }

    /// Test hook: the local pool order (key per pool index) of a submap.
    #[cfg(test)]
    pub(crate) fn pool_keys(&self, id: SubmapId) -> Option<&[u64]> {
        self.slot_of(id)
            .map(|s| self.headers[s].index.pool_keys.as_slice())
    }

    /// Blocks of a submap.
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// Its allocated block count, `None` for an unknown id.
    pub fn submap_blocks(&self, id: SubmapId) -> Option<usize> {
        self.slot_of(id).map(|s| self.headers[s].index.len())
    }

    /// Extracts the meshes of the selected submaps in the world frame,
    /// ascending submap id (each as [`super::TsdfVolume::extract_mesh_into`],
    /// mapped by its `world_from_submap`).
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `filter` - Submaps to mesh.
    /// * `params` - Weight threshold, normals, colours.
    /// * `mesh` - Output mesh (cleared first).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the surface.
    ///
    /// # Errors
    ///
    /// [`TsdfError::UnknownAgent`], [`TsdfError::UnknownSubmap`],
    /// [`TsdfError::InvalidParameter`], [`TsdfError::MeshTooLarge`],
    /// [`TsdfError::FpEnvironment`].
    pub fn extract_mesh_into(
        &self,
        ws: &mut TsdfWorkspace,
        filter: SubmapFilter<'_>,
        params: &MeshParams,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        check_fp_env()?;
        params.validate(self.arena.color)?;
        self.check_filter(&filter)?;
        mesh.clear();
        for &slot in &self.live {
            let h = &self.headers[slot as usize];
            if filter.matches(h.id) {
                let xf = rigid_from_pose(&h.world_from_submap)?;
                extract_view(&self.view(slot as usize), params, ws, mesh, Some(&xf))?;
            }
        }
        Ok(())
    }

    /// Re-meshes the changed blocks of every submap into `cache`.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `params` - Weight threshold, normals, colours.
    /// * `cache` - This map's cache.
    ///
    /// # Returns
    ///
    /// Summed counters over the submaps.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`], [`TsdfError::FpEnvironment`].
    pub fn update_mesh_cache(
        &self,
        ws: &mut TsdfWorkspace,
        params: &MeshParams,
        cache: &mut SubmapMeshCache,
    ) -> Result<MeshUpdateStats, TsdfError> {
        check_fp_env()?;
        params.validate(self.arena.color)?;
        let mut stats = MeshUpdateStats::default();
        for &slot in &self.live {
            let s = slot as usize;
            if cache.caches.len() <= s {
                cache.caches.resize_with(s + 1, BlockMeshCache::default);
                cache.ids.resize(s + 1, None);
            }
            let id = self.headers[s].id;
            if cache.ids[s] != Some(id) {
                cache.caches[s].invalidate();
                cache.ids[s] = Some(id);
            }
            let st = update_cache(&self.view(s), ws, params, &mut cache.caches[s]);
            stats.accumulate(&st);
        }
        Ok(stats)
    }

    fn collect_rays(
        &self,
        ws: &mut TsdfWorkspace,
        camera: &PinholeCamera,
        size: ImageSize,
        world_from_camera: &Pose3d,
        filter: &SubmapFilter<'_>,
        params: &RaycastParams,
    ) -> Result<(), TsdfError> {
        let k = check_raycast(camera, size, params, self.arena.color)?;
        check_pose(world_from_camera)?;
        self.check_filter(filter)?;
        ws.ray_params.clear();
        ws.ray_slots.clear();
        for &slot in &self.live {
            let h = &self.headers[slot as usize];
            if !filter.matches(h.id) {
                continue;
            }
            let submap_from_camera = h.world_from_submap.inverse().compose(world_from_camera);
            let rp = core_params(
                &self.params.layout,
                Self::LOG2,
                &k,
                &submap_from_camera,
                params,
            )?;
            ws.ray_params.push(rp);
            ws.ray_slots.push(slot);
        }
        Ok(())
    }

    fn cast(&self, ws: &mut TsdfWorkspace, size: ImageSize, out: RayOut<'_>) {
        let rps = std::mem::take(&mut ws.ray_params);
        let slots = std::mem::take(&mut ws.ray_slots);
        raycast_views(size, &rps, &|i| self.view(slots[i] as usize), out);
        ws.ray_params = rps;
        ws.ray_slots = slots;
    }

    /// Raycasts a pinhole view of the selected submaps: each submap is
    /// marched in its own frame, ascending id, and the smallest hit depth
    /// wins (the lower id on a tie). Outputs are in the camera frame.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `camera` - Intrinsics of the view, without distortion.
    /// * `world_from_camera` - Camera pose in the world frame.
    /// * `filter` - Submaps to raycast.
    /// * `params` - March parameters.
    /// * `maps` - Output maps.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the maps hold the view.
    ///
    /// # Errors
    ///
    /// As [`super::TsdfVolume::raycast_into`], plus
    /// [`TsdfError::UnknownAgent`] and [`TsdfError::UnknownSubmap`].
    pub fn raycast_into(
        &self,
        ws: &mut TsdfWorkspace,
        camera: &PinholeCamera,
        world_from_camera: &Pose3d,
        filter: SubmapFilter<'_>,
        params: &RaycastParams,
        maps: &mut RaycastMaps,
    ) -> Result<(), TsdfError> {
        let size = maps.size();
        self.collect_rays(ws, camera, size, world_from_camera, &filter, params)?;
        let out = maps_out(size, params, maps)?;
        self.cast(ws, size, out);
        Ok(())
    }

    /// Raycasts every level of an RGB-D ICP pyramid (vertex and normal
    /// maps at each level's camera) from the selected submaps.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `world_from_camera` - Camera pose in the world frame.
    /// * `filter` - Submaps to raycast.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `pyramid` - Pyramid to refill.
    ///
    /// # Returns
    ///
    /// `Ok(())` once every level holds the view.
    ///
    /// # Errors
    ///
    /// As [`SubmapTsdf::raycast_into`].
    pub fn raycast_pyramid_into(
        &self,
        ws: &mut TsdfWorkspace,
        world_from_camera: &Pose3d,
        filter: SubmapFilter<'_>,
        params: &RaycastParams,
        pyramid: &mut RgbdPyramid,
    ) -> Result<(), TsdfError> {
        let params = RaycastParams {
            colors: false,
            ..*params
        };
        for level in pyramid.levels.iter_mut() {
            let size = level.size();
            let camera = level.camera().clone();
            self.collect_rays(ws, &camera, size, world_from_camera, &filter, &params)?;
            self.cast(ws, size, level_out(level)?);
        }
        Ok(())
    }

    /// Memory held.
    ///
    /// # Arguments
    ///
    /// * `agent` - One agent's submaps (their indices and voxel pages), or
    ///   `None` for everything the map holds (arena and slot capacity).
    ///
    /// # Returns
    ///
    /// Host bytes, device bytes (0) and allocated blocks.
    pub fn memory_bytes(&self, agent: Option<AgentId>) -> MemoryReport {
        let mut blocks = 0;
        let mut host = 0;
        for &slot in &self.live {
            let h = &self.headers[slot as usize];
            if agent.is_none_or(|a| a == h.id.agent) {
                blocks += h.index.len();
                if agent.is_some() {
                    host += h.index.bytes() + self.arena.page_bytes(h.index.len());
                }
            }
        }
        if agent.is_none() {
            host = self.arena.bytes() + self.headers.iter().map(|h| h.index.bytes()).sum::<usize>();
        }
        MemoryReport {
            host_bytes: host,
            device_bytes: 0,
            blocks,
        }
    }

    /// The submaps a device-resident copy keeps under
    /// [`SubmapParams::max_resident_bytes`]: every active submap, then the
    /// others from the most recently integrated (LRU) while their voxel
    /// pages fit the budget; all submaps without a budget. Ascending id.
    ///
    /// # Arguments
    ///
    /// * `out` - Output ids (cleared first).
    pub fn resident_submaps_into(&self, out: &mut Vec<SubmapId>) {
        out.clear();
        let Some(budget) = self.params.max_resident_bytes else {
            out.extend(self.submap_ids());
            return;
        };
        let bytes = |slot: u32| {
            self.arena
                .page_bytes(self.headers[slot as usize].index.len())
        };
        let active = |slot: u32| self.agents.iter().any(|a| a.active == Some(slot));
        let mut used: usize = self
            .live
            .iter()
            .filter(|&&s| active(s))
            .map(|&s| bytes(s))
            .sum();
        out.extend(
            self.live
                .iter()
                .filter(|&&s| active(s))
                .map(|&s| self.headers[s as usize].id),
        );
        // most recent first: last_used is unique per step
        let mut last = u64::MAX;
        loop {
            let next = self
                .live
                .iter()
                .filter(|&&s| !active(s) && self.headers[s as usize].last_used < last)
                .max_by_key(|&&s| self.headers[s as usize].last_used);
            let Some(&s) = next else { break };
            last = self.headers[s as usize].last_used;
            if used + bytes(s) > budget {
                break;
            }
            used += bytes(s);
            out.push(self.headers[s as usize].id);
        }
        out.sort_unstable();
    }

    /// FNV-1a 64 of one submap's blocks (as
    /// [`super::TsdfVolume::content_hash`]).
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// The hash, `None` for an unknown id.
    pub fn content_hash(&self, id: SubmapId) -> Option<u64> {
        self.slot_of(id)
            .map(|s| content_hash(&self.headers[s].index, &self.arena))
    }
}
