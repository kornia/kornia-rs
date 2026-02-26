//! # Lie Groups
//!
//! This module provides implementations of Lie groups used in computer vision and robotics.
//!
//! ## What is a Lie group?
//!
//! A Lie group is a group that is also a smooth manifold — meaning it has both algebraic
//! structure (composition, inverse, identity) and geometric structure (tangent spaces,
//! smooth curves). This combination is what makes Lie groups the natural language for
//! describing continuous transformations like rotations and rigid body motions.
//!
//! Every Lie group has a corresponding **Lie algebra**: the tangent space at the identity
//! element. The Lie algebra is a plain vector space, which makes it suitable for
//! optimization (gradient descent, least squares). The **exponential map** (`exp`) moves
//! from the Lie algebra to the group, and the **logarithmic map** (`log`) goes back.
//!
//! ## Implemented groups
//!
//! | Group | Description | DOF | Topology | Internal repr |
//! |-------|-------------|-----|----------|---------------|
//! | [`SO2F32`] | 2D rotations | 1 | S¹ (circle) | Complex number |
//! | [`SE2F32`] | 2D rigid body transforms | 3 | SO(2) ⋉ R² | SO2 + Vec2 |
//! | [`SO3F32`] | 3D rotations | 3 | RP³ (quotient of S³) | Unit quaternion |
//! | [`SE3F32`] | 3D rigid body transforms | 6 | SO(3) ⋉ R³ | SO3 + Vec3 |
//! | [`RxSO3F32`] | 3D rotation + scale | 4 | R⁺ × SO(3) | Scaled quaternion |
//! | [`Sim3F32`] | 3D similarity transforms | 7 | (R⁺ × SO(3)) ⋉ R³ | RxSO3 + Vec3 |
//!
//! ## Group relationships
//!
//! ```text
//!                    Spin group (double cover)
//!   Unit quaternions ≅ SU(2) ≅ S³
//!                      │
//!                      │ 2:1 quotient map (q and -q map to same rotation)
//!                      ▼
//!                    SO(3) ≅ RP³          ← SO3F32 lives here
//!                      │
//!                      │ semi-direct product with R³
//!                      ▼
//!                    SE(3) = SO(3) ⋉ R³   ← SE3F32 lives here
//!                      │
//!                      │ add scale (R⁺)
//!                      ▼
//!                    Sim(3) = (R⁺ × SO(3)) ⋉ R³  ← Sim3F32 lives here
//! ```
//!
//! ## The double cover: SU(2) → SO(3)
//!
//! [`SO3F32`] is stored internally as a unit quaternion, but a unit quaternion is actually
//! an element of SU(2), the double cover of SO(3). The key consequence: **`q` and `-q`
//! represent the same rotation**. SO(3) is the quotient SU(2)/{±1} — every rotation
//! corresponds to a pair of antipodal quaternions on the 3-sphere S³.
//!
//! This is not a bug or an implementation detail — it reflects the topology of the
//! rotation group. SO(3) is topologically RP³ (real projective 3-space), which is not
//! simply connected: a 360° rotation is a non-contractible loop. SU(2) ≅ S³ is its
//! universal cover, where you need 720° to close a loop. The objects that are sensitive
//! to this distinction — that pick up a sign under 360° rotation — are called **spinors**.
//!
//! Practical consequences:
//! - [`SO3F32::from_matrix`] must choose one of two quaternions — there is no
//!   globally continuous way to make this choice over all of SO(3).
//! - Quaternion interpolation (SLERP) is smooth because it operates in SU(2) ≅ S³.
//! - When comparing quaternions, check both `q ≈ q'` and `q ≈ -q'`.
//! - Repeated quaternion multiplications accumulate floating point drift off S³;
//!   call [`QuatF32::normalize`] periodically to project back onto the unit sphere.
//!
//! ## Quotients as a recurring pattern
//!
//! The SU(2) → SO(3) double cover is an instance of a general pattern: a **quotient**
//! collapses a larger space by identifying equivalent points. This same pattern appears
//! in two-view geometry: the essential manifold is a quotient of SE(3) where the map
//! `(R, t) → [t]× R` collapses all translations along the same direction (losing scale
//! and sign). In both cases, the quotient is smaller and simpler, but loses information
//! that can only be recovered by additional constraints (cheirality checks, sign
//! conventions).
//!
//! ## Common API
//!
//! All Lie group types implement:
//! - `exp(v)` — exponential map: Lie algebra → group
//! - `log()` — logarithmic map: group → Lie algebra
//! - `hat(v)` — vector → matrix representation of Lie algebra element
//! - `vee(M)` — matrix → vector (inverse of hat)
//! - `adjoint()` — adjoint representation (how the group acts on its own Lie algebra)
//! - `inverse()` — group inverse
//! - `*` operator — group composition
//! - `rplus(tau)` / `lplus(tau)` — retraction from tangent space (right/left)
//! - `rminus(&other)` / `lminus(&other)` — logarithmic difference
//! - `left_jacobian(v)` / `right_jacobian(v)` — analytical Jacobians for optimization
//! - `matrix()` — convert to homogeneous matrix representation

pub mod rxso3;
pub mod se2;
pub mod se3;
pub mod sim3;
pub mod so2;
pub mod so3;

// Re-export types at module root for convenience
pub use rxso3::RxSO3F32;
pub use se2::SE2F32;
pub use se3::SE3F32;
pub use sim3::Sim3F32;
pub use so2::SO2F32;
pub use so3::SO3F32;
