//! Correctness verification for both decoders.
//!
//! Verifies that corrections properly resolve all defects by
//! XOR-toggling syndrome bits at edge endpoints.

use fusion_blossom::util::{EdgeIndex, VertexIndex, Weight};
use prav_core::EdgeCorrection;

/// Verify prav corrections resolve all defects.
///
/// Applies corrections by XOR-toggling syndrome bits at edge endpoints.
/// A valid matching should resolve all defects to zero.
///
/// Returns (all_resolved, original_defects, remaining_defects).
pub fn verify_prav_corrections(
    syndromes: &[u64],
    corrections: &[EdgeCorrection],
    _width: usize,
    _height: usize,
) -> (bool, usize, usize) {
    let original = syndromes.iter().map(|x| x.count_ones() as usize).sum();

    if original == 0 {
        return (true, 0, 0);
    }

    let mut result = syndromes.to_vec();

    for EdgeCorrection { u, v } in corrections {
        // Toggle bit at u (boundary marker = u32::MAX)
        if *u != u32::MAX {
            let block = *u as usize / 64;
            let bit = *u as usize % 64;
            if block < result.len() {
                result[block] ^= 1u64 << bit;
            }
        }

        // Toggle bit at v (boundary marker = u32::MAX)
        if *v != u32::MAX {
            let block = *v as usize / 64;
            let bit = *v as usize % 64;
            if block < result.len() {
                result[block] ^= 1u64 << bit;
            }
        }
    }

    let remaining: usize = result.iter().map(|x| x.count_ones() as usize).sum();
    (remaining == 0, original, remaining)
}

/// Verify fusion-blossom corrections resolve all defects.
///
/// Reconstructs syndrome change from selected edges and verifies
/// that applying the change resolves all defects.
///
/// Returns (all_resolved, original_defects, remaining_defects).
pub fn verify_fb_corrections(
    defects: &[VertexIndex],
    edge_corrections: &[EdgeIndex],
    edges: &[(VertexIndex, VertexIndex, Weight)],
    width: usize,
    height: usize,
) -> (bool, usize, usize) {
    let original = defects.len();

    if original == 0 {
        return (true, 0, 0);
    }

    let num_nodes = width * height;

    // Build syndrome change from selected edges
    let mut syndrome_change = vec![false; num_nodes];

    for &edge_idx in edge_corrections {
        if (edge_idx as usize) < edges.len() {
            let (u, v, _) = edges[edge_idx as usize];

            // Toggle at u if it's a real node (not boundary)
            if (u as usize) < num_nodes {
                syndrome_change[u as usize] ^= true;
            }

            // Toggle at v if it's a real node (not boundary)
            if (v as usize) < num_nodes {
                syndrome_change[v as usize] ^= true;
            }
        }
    }

    // Build original syndrome as a set
    let mut original_syndrome = vec![false; num_nodes];
    for &d in defects {
        if (d as usize) < num_nodes {
            original_syndrome[d as usize] = true;
        }
    }

    // Apply change (XOR) and count remaining
    let mut remaining = 0;
    for i in 0..num_nodes {
        let after = original_syndrome[i] ^ syndrome_change[i];
        if after {
            remaining += 1;
        }
    }

    (remaining == 0, original, remaining)
}
