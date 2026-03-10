//! Linear algebra operations for kornia-algebra
//!
//! This module provides higher-level linear algebra algorithms that operate on
//! algebraic types.

/// Rigid body alignment utilities (Kabsch / Umeyama)
pub mod rigid;

/// Singular Value Decomposition for 3x3 matrices
pub mod svd;

/// Cholesky Decomposition for 3x3 matrices
pub mod cholesky;
