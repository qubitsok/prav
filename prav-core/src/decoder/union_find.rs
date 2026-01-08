//! Union Find (Disjoint Set Forest) implementation for cluster tracking.
//!
//! This module provides the data structure that tracks which nodes belong to
//! the same cluster during QEC decoding. It uses several optimizations:
//!
//! - **Fast path for self-rooted nodes**: At typical error rates (p=0.001),
//!   ~95% of nodes are self-rooted (their own cluster). Direct check avoids traversal.
//! - **Path halving compression**: When traversing, each node points to its grandparent,
//!   halving path length per query. Achieves O(α(n)) amortized complexity.
//! - **Deterministic union**: Smaller index becomes child of larger, providing
//!   reproducible results without rank tracking overhead.

#![allow(unsafe_op_in_unsafe_fn)]
use super::state::DecodingState;
use crate::topology::Topology;

/// Disjoint set forest operations for tracking connected clusters.
///
/// In Union Find-based QEC decoding, each syndrome node starts as its own cluster.
/// As cluster growth proceeds, neighboring nodes are merged using `union`. The
/// `find` operation determines which cluster a node belongs to.
///
/// # Cluster Representation
///
/// Each cluster is identified by its root node - the representative element.
/// The root is the node where `parents[root] == root`. All other nodes in the
/// cluster have a parent pointer forming a tree structure leading to the root.
///
/// ```text
/// Before union:       After union(A, B):
///   A    B               B (root)
///  /|    |              /|\
/// 1 2    3             A 1 2
///                        |
///                        3
/// ```
///
/// # Performance Characteristics
///
/// | Operation | Time Complexity | Notes |
/// |-----------|-----------------|-------|
/// | `find` | O(α(n)) amortized | α is inverse Ackermann (effectively constant) |
/// | `union` | O(α(n)) amortized | Two finds + O(1) merge |
/// | `union_roots` | O(1) | Direct merge of known roots |
pub trait UnionFind {
    /// Finds the root (cluster representative) of the node `i`.
    ///
    /// This is the fundamental query operation. Two nodes are in the same cluster
    /// if and only if they have the same root.
    ///
    /// # Arguments
    ///
    /// * `i` - Node index to find the root of.
    ///
    /// # Returns
    ///
    /// The root node index of the cluster containing `i`.
    ///
    /// # Fast Path Optimization
    ///
    /// At typical QEC error rates, most nodes are isolated (self-rooted).
    /// The implementation checks `parents[i] == i` first, returning immediately
    /// in ~95% of cases without any traversal.
    ///
    /// # Path Compression
    ///
    /// During traversal, path halving is applied: each visited node is redirected
    /// to its grandparent. This flattens the tree over time, keeping paths short.
    fn find(&mut self, i: u32) -> u32;

    /// Merges two clusters given their root nodes.
    ///
    /// This is the low-level merge operation used when roots are already known.
    /// Use [`union`](Self::union) for the general case where roots must be found.
    ///
    /// # Arguments
    ///
    /// * `root_u` - Root of the first cluster.
    /// * `root_v` - Root of the second cluster.
    ///
    /// # Returns
    ///
    /// * `true` if the clusters were merged (they were different).
    /// * `false` if they were already the same cluster.
    ///
    /// # Union Strategy
    ///
    /// Uses index-based union: the smaller index becomes a child of the larger.
    /// This provides deterministic behavior without maintaining rank information.
    ///
    /// # Safety
    ///
    /// Caller must ensure `root_u` and `root_v` are valid root node indices
    /// (i.e., `parents[root_u] == root_u` and `parents[root_v] == root_v`).
    unsafe fn union_roots(&mut self, root_u: u32, root_v: u32) -> bool;

    /// Merges the clusters containing nodes `u` and `v`.
    ///
    /// Finds the roots of both nodes and merges them if different.
    ///
    /// # Arguments
    ///
    /// * `u` - First node index.
    /// * `v` - Second node index.
    ///
    /// # Returns
    ///
    /// * `true` if the clusters were merged.
    /// * `false` if `u` and `v` were already in the same cluster.
    ///
    /// # Safety
    ///
    /// Caller must ensure `u` and `v` are valid node indices within bounds.
    unsafe fn union(&mut self, u: u32, v: u32) -> bool;
}

impl<'a, T: Topology, const STRIDE_Y: usize> UnionFind for DecodingState<'a, T, STRIDE_Y> {
    // Optimized find with O(1) fast path for self-rooted nodes
    // At p=0.001, ~95% of nodes are self-rooted, so this check pays off
    #[inline(always)]
    fn find(&mut self, i: u32) -> u32 {
        // SAFETY: Callers must ensure `i < parents.len()`. This is an internal
        // method called only from growth and peeling code that iterates over
        // valid node indices. The unchecked access eliminates bounds checking
        // in the hot path.
        unsafe {
            let p = *self.parents.get_unchecked(i as usize);
            if p == i {
                return i; // Fast path: self-rooted (most common case)
            }
            self.find_slow(i, p)
        }
    }

    #[inline(always)]
    unsafe fn union_roots(&mut self, root_u: u32, root_v: u32) -> bool {
        if root_u == root_v {
            return false;
        }

        // Simple index-based union: smaller index becomes child of larger
        // This provides deterministic behavior without rank tracking overhead
        let (child, parent) = if root_u < root_v {
            (root_u, root_v)
        } else {
            (root_v, root_u)
        };

        // SAFETY: Caller guarantees root_u and root_v are valid root indices
        // (i.e., `parents[root] == root`). Since they're roots, they're valid
        // node indices by construction.
        *self.parents.get_unchecked_mut(child as usize) = parent;

        // Invalidate cached root for the child's block
        let blk_child = (child as usize) >> 6;
        if blk_child < self.blocks_state.len() {
            // SAFETY: Bounds check performed above.
            self.blocks_state.get_unchecked_mut(blk_child).root = u32::MAX;
        }
        self.mark_block_dirty(blk_child);

        true
    }

    #[inline(always)]
    unsafe fn union(&mut self, u: u32, v: u32) -> bool {
        // SAFETY: Caller guarantees u and v are valid node indices.
        // find() performs unchecked access but is safe given valid indices.
        let root_u = self.find(u);
        let root_v = self.find(v);

        if root_u != root_v {
            // SAFETY: root_u and root_v are valid roots from find().
            self.union_roots(root_u, root_v)
        } else {
            false
        }
    }
}

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    // Cold path: path halving compression
    // Each node on the path points to its grandparent, halving path length per traversal.
    #[inline(never)]
    #[cold]
    fn find_slow(&mut self, mut i: u32, mut p: u32) -> u32 {
        // SAFETY: This function is only called from find() with valid indices.
        // The parent pointers form a well-formed tree structure where every
        // node either points to itself (root) or to a valid parent index.
        // The loop terminates when we reach a root (p == grandparent).
        unsafe {
            loop {
                let grandparent = *self.parents.get_unchecked(p as usize);
                if p == grandparent {
                    return p; // Found root
                }
                // Path halving: point i to grandparent
                *self.parents.get_unchecked_mut(i as usize) = grandparent;
                self.mark_block_dirty(i as usize >> 6);
                i = grandparent;
                p = *self.parents.get_unchecked(i as usize);
            }
        }
    }
}
