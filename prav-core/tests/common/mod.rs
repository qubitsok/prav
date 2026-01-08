//! Shared test utilities for prav-core tests.
//!
//! This module provides common helper functions used across multiple test files
//! to verify decoder correctness and calculate node indices.

#![allow(dead_code)] // Not all test files use all functions

use prav_core::decoder::EdgeCorrection;

/// Verifies that applying corrections to defects results in zero remaining defects.
///
/// This function applies corrections by XOR-toggling the endpoints and checks
/// if all defects are resolved.
///
/// Returns `Ok(())` if all defects are matched, `Err` with remaining defect indices otherwise.
pub fn verify_matching(dense_input: &[u64], corrections: &[EdgeCorrection]) -> Result<(), Vec<usize>> {
    let mut state = dense_input.to_vec();

    // Toggle correction endpoints
    for c in corrections {
        let u = c.u as usize;
        let v = c.v as usize;

        let u_blk = u / 64;
        let u_bit = u % 64;
        if u_blk < state.len() {
            state[u_blk] ^= 1 << u_bit;
        }

        // v might be u32::MAX (boundary), which will be out of bounds
        // and safely ignored, matching the "open boundary" behavior.
        let v_blk = v / 64;
        let v_bit = v % 64;
        if v_blk < state.len() {
            state[v_blk] ^= 1 << v_bit;
        }
    }

    // Collect remaining defects
    let mut remaining = Vec::new();
    for (blk_idx, &word) in state.iter().enumerate() {
        if word != 0 {
            let mut w = word;
            let base = blk_idx * 64;
            while w != 0 {
                let b = w.trailing_zeros();
                w &= w - 1;
                remaining.push(base + b as usize);
            }
        }
    }

    if remaining.is_empty() {
        Ok(())
    } else {
        Err(remaining)
    }
}

/// Verifies matching and returns a boolean (for proptest assertions).
///
/// Returns `true` if all defects are resolved, `false` otherwise.
#[inline]
pub fn verify_matching_bool(dense_input: &[u64], corrections: &[EdgeCorrection]) -> bool {
    verify_matching(dense_input, corrections).is_ok()
}

/// Calculate linear index from 2D coordinates for a square grid.
///
/// Uses power-of-two stride for efficient indexing.
#[inline]
pub fn idx(x: usize, y: usize, w: usize, h: usize) -> u32 {
    let max_dim = w.max(h);
    let stride_y = max_dim.next_power_of_two();
    ((y * stride_y) + x) as u32
}
