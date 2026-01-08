//! Peeling decoder for extracting corrections from grown clusters.
//!
//! After cluster growth completes, the peeling phase extracts the actual
//! error corrections. The algorithm traces paths through the Union Find
//! forest to identify which edges should be corrected.
//!
//! # Peeling Algorithm
//!
//! The peeling algorithm works by:
//!
//! 1. **Forest construction**: The Union Find parent pointers form a spanning forest
//! 2. **Path tracing**: For each defect, trace the path to the cluster root
//! 3. **XOR cancellation**: Paths that overlap cancel out (XOR operation)
//! 4. **Edge extraction**: Remaining edges form the correction
//!
//! # Correctness
//!
//! The key insight is that applying corrections along the traced paths will
//! flip the syndrome at exactly the right locations to cancel all defects.
//! Overlapping paths cancel via XOR, leaving only the minimal correction.
//!
//! # Path Tracing Strategies
//!
//! Multiple strategies are used depending on the situation:
//!
//! - **BFS within block**: For paths contained in a single 64-node block
//! - **Manhattan path**: For paths crossing multiple blocks
//! - **Bitmask BFS**: Optimized for small grids using bitmask operations

#![allow(unsafe_op_in_unsafe_fn)]

use crate::decoder::state::DecodingState;
use crate::decoder::types::EdgeCorrection;
use crate::intrinsics::tzcnt;
use crate::topology::Topology;

/// Correction extraction operations for QEC decoding.
///
/// After clusters have grown to completion, this trait provides the operations
/// to extract the actual edge corrections that should be applied to fix errors.
///
/// # Output Format
///
/// Corrections are returned as a list of [`EdgeCorrection`] structures:
/// - Internal edges: `EdgeCorrection { u, v }` where both are valid node indices
/// - Boundary edges: `EdgeCorrection { u, v: u32::MAX }` indicating boundary match
///
/// # Usage
///
/// ```ignore
/// // After loading syndromes and growing clusters
/// let mut corrections = vec![EdgeCorrection::default(); max_corrections];
/// let num_corrections = state.peel_forest(&mut corrections);
///
/// // Apply corrections to physical qubits
/// for i in 0..num_corrections {
///     apply_correction(corrections[i]);
/// }
/// ```
pub trait Peeling {
    /// Performs full decoding: cluster growth followed by peeling.
    ///
    /// This is a convenience method that calls [`grow_clusters`](crate::ClusterGrowth::grow_clusters)
    /// followed by [`peel_forest`](Self::peel_forest).
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections.
    ///
    /// # Returns
    ///
    /// Number of corrections written to the buffer.
    fn decode(&mut self, corrections: &mut [EdgeCorrection]) -> usize;

    /// Extracts corrections from the grown cluster forest.
    ///
    /// Traces paths from each defect node through the Union Find forest,
    /// marking edges along the way. Overlapping paths cancel via XOR.
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections. Should be large
    ///   enough to hold all possible corrections (typically `num_nodes / 2`).
    ///
    /// # Returns
    ///
    /// Number of corrections written to the buffer.
    ///
    /// # Algorithm
    ///
    /// 1. Clear path marks
    /// 2. For each defect node, trace path to root, XOR-marking each node
    /// 3. For each marked node pair, trace Manhattan path between them
    /// 4. Extract edges from the edge bitmaps
    fn peel_forest(&mut self, corrections: &mut [EdgeCorrection]) -> usize;

    /// Converts marked edges to EdgeCorrection structures.
    ///
    /// Scans the edge and boundary bitmaps populated during path tracing,
    /// extracting them into the corrections output array.
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections.
    ///
    /// # Returns
    ///
    /// Number of corrections written to the buffer.
    fn reconstruct_corrections(&mut self, corrections: &mut [EdgeCorrection]) -> usize;

    /// Traces a path from node `u` toward the boundary through parent pointers.
    ///
    /// XOR-marks each node along the path in `path_mark`. When two paths
    /// overlap, the XOR cancels, leaving only the symmetric difference.
    ///
    /// # Arguments
    ///
    /// * `u` - Starting node for path tracing.
    /// * `boundary_node` - The boundary sentinel node index.
    fn trace_path(&mut self, u: u32, boundary_node: u32);

    /// Traces a path using BFS within a single block.
    ///
    /// Used when both endpoints are in the same 64-node block. BFS finds
    /// the shortest path through occupied nodes.
    ///
    /// # Arguments
    ///
    /// * `u` - Start node.
    /// * `v` - End node.
    /// * `mask` - Bitmask of valid/occupied nodes in the block.
    fn trace_bfs(&mut self, u: u32, v: u32, mask: u64);

    /// Traces paths using bitmask-based BFS for small grids.
    ///
    /// Optimized for grids with <=64 blocks (<=4096 nodes). Uses bitmask
    /// operations for efficient visited tracking and queue management.
    ///
    /// # Arguments
    ///
    /// * `start_node` - The defect node to trace from.
    fn trace_bitmask_bfs(&mut self, start_node: u32);

    /// Traces a Manhattan (axis-aligned) path between two nodes.
    ///
    /// Used for paths crossing block boundaries. Moves along X, then Y,
    /// then Z axes, emitting edge corrections along the way.
    ///
    /// # Arguments
    ///
    /// * `u` - Start node.
    /// * `v` - End node.
    fn trace_manhattan(&mut self, u: u32, v: u32);

    /// Emits a single edge correction between two nodes.
    ///
    /// Records the edge in the appropriate bitmap (edge or boundary).
    /// Uses XOR so overlapping emissions cancel out.
    ///
    /// # Arguments
    ///
    /// * `u` - First endpoint.
    /// * `v` - Second endpoint, or `u32::MAX` for boundary edge.
    fn emit_linear(&mut self, u: u32, v: u32);

    /// Converts a linear node index to (x, y, z) coordinates.
    ///
    /// Uses precomputed shifts for efficient division.
    ///
    /// # Arguments
    ///
    /// * `u` - Node index.
    ///
    /// # Returns
    ///
    /// Tuple `(x, y, z)` coordinates. For 2D grids, `z` is always 0.
    fn get_coord(&self, u: u32) -> (usize, usize, usize);
}

impl<'a, T: Topology, const STRIDE_Y: usize> Peeling for DecodingState<'a, T, STRIDE_Y> {
    fn decode(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        self.grow_clusters();
        self.peel_forest(corrections)
    }

    fn peel_forest(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        self.path_mark.fill(0);
        let boundary_node = (self.parents.len() - 1) as u32;
        let _is_small_grid = STRIDE_Y <= 32 && self.blocks_state.len() <= 17;

        for blk_idx in 0..self.defect_mask.len() {
            let mut word = unsafe { *self.defect_mask.get_unchecked(blk_idx) };
            if word == 0 {
                continue;
            }

            let base_node = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_node + bit) as u32;

                if _is_small_grid {
                    let root = self.find(u);
                    if root == u && root != boundary_node {
                        let occ = unsafe { self.blocks_state.get_unchecked(blk_idx).occupied };
                        if (occ & (1 << bit)) != 0 {
                            self.trace_bitmask_bfs(u);
                            continue;
                        }
                    }
                }

                self.trace_path(u, boundary_node);
            }
        }

        for blk_idx in 0..self.path_mark.len() {
            let mut word = unsafe { *self.path_mark.get_unchecked(blk_idx) };
            if word == 0 {
                continue;
            }

            let base_node = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_node + bit) as u32;
                let v = unsafe { *self.parents.get_unchecked(u as usize) };

                if u != v {
                    self.trace_manhattan(u, v);
                }
            }
        }

        self.reconstruct_corrections(corrections)
    }

    fn reconstruct_corrections(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        let mut count = 0;

        for i in 0..self.edge_dirty_count {
            let word_idx = unsafe { *self.edge_dirty_list.get_unchecked(i) } as usize;

            // Clear mask bit
            let mask_idx = word_idx >> 6;
            let mask_bit = word_idx & 63;
            unsafe {
                *self.edge_dirty_mask.get_unchecked_mut(mask_idx) &= !(1 << mask_bit);
            }

            let word_ptr = unsafe { self.edge_bitmap.get_unchecked_mut(word_idx) };
            let mut word = *word_ptr;
            *word_ptr = 0;

            let base_idx = word_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;

                let global_idx = base_idx + bit;
                let u = (global_idx / 3) as u32;
                let dir = global_idx % 3;

                let v = match dir {
                    0 => u + 1,
                    1 => u + self.stride_y as u32,
                    2 => u + self.graph.stride_z as u32,
                    _ => unsafe { core::hint::unreachable_unchecked() },
                };

                if count < corrections.len() {
                    unsafe {
                        *corrections.get_unchecked_mut(count) = EdgeCorrection { u, v };
                    }
                    count += 1;
                }
            }
        }
        self.edge_dirty_count = 0;

        for i in 0..self.boundary_dirty_count {
            let blk_idx = unsafe { *self.boundary_dirty_list.get_unchecked(i) } as usize;

            // Clear mask bit
            let mask_idx = blk_idx >> 6;
            let mask_bit = blk_idx & 63;
            unsafe {
                *self.boundary_dirty_mask.get_unchecked_mut(mask_idx) &= !(1 << mask_bit);
            }

            let word_ptr = unsafe { self.boundary_bitmap.get_unchecked_mut(blk_idx) };
            let mut word = *word_ptr;
            *word_ptr = 0;

            let base_u = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_u + bit) as u32;
                if count < corrections.len() {
                    unsafe {
                        *corrections.get_unchecked_mut(count) = EdgeCorrection { u, v: u32::MAX };
                    }
                    count += 1;
                }
            }
        }
        self.boundary_dirty_count = 0;
        count
    }

    fn trace_path(&mut self, u: u32, _boundary_node: u32) {
        let mut curr = u;
        loop {
            let next = unsafe { *self.parents.get_unchecked(curr as usize) };

            if curr == next {
                break;
            }

            let blk = (curr as usize) / 64;
            let bit = (curr as usize) % 64;
            let mark_ptr = unsafe { self.path_mark.get_unchecked_mut(blk) };
            let mask = 1 << bit;

            *mark_ptr ^= mask;

            curr = next;
        }
    }

    fn trace_bfs(&mut self, u: u32, v: u32, mask: u64) {
        if u == v {
            return;
        }

        let u_local = (u % 64) as usize;
        let v_local = (v % 64) as usize;

        let mut pred = [64u8; 64];
        let mut visited = 0u64;
        let mut queue = 0u64;

        visited |= 1 << u_local;
        queue |= 1 << u_local;
        pred[u_local] = u_local as u8;

        let stride_y = self.stride_y;
        let do_vertical = stride_y < 64;

        let mut found = false;

        while queue != 0 {
            let curr_bit = tzcnt(queue) as usize;
            queue &= queue - 1;

            if curr_bit == v_local {
                found = true;
                break;
            }

            if curr_bit > 0 && (self.row_start_mask & (1 << curr_bit)) == 0 {
                try_queue(
                    curr_bit - 1,
                    curr_bit,
                    mask,
                    &mut visited,
                    &mut queue,
                    &mut pred,
                );
            }
            if curr_bit < 63 && (self.row_end_mask & (1 << curr_bit)) == 0 {
                try_queue(
                    curr_bit + 1,
                    curr_bit,
                    mask,
                    &mut visited,
                    &mut queue,
                    &mut pred,
                );
            }
            if do_vertical {
                if curr_bit >= stride_y {
                    try_queue(
                        curr_bit - stride_y,
                        curr_bit,
                        mask,
                        &mut visited,
                        &mut queue,
                        &mut pred,
                    );
                }
                if curr_bit + stride_y < 64 {
                    try_queue(
                        curr_bit + stride_y,
                        curr_bit,
                        mask,
                        &mut visited,
                        &mut queue,
                        &mut pred,
                    );
                }
            }
        }

        if found {
            let mut curr = v_local;
            let base = (u / 64) * 64;
            while curr != u_local {
                let p = pred[curr] as usize;
                let u_abs = base + p as u32;
                let v_abs = base + curr as u32;
                self.emit_linear(u_abs, v_abs);
                curr = p;
            }
        } else {
            self.trace_manhattan(u, v);
        }
    }

    fn trace_bitmask_bfs(&mut self, start_node: u32) {
        if STRIDE_Y <= 32 {
            let mut visited = [0u64; 17];
            self.trace_bitmask_bfs_impl(start_node, &mut visited);
        } else {
            let mut visited = [0u64; 65];
            self.trace_bitmask_bfs_impl(start_node, &mut visited);
        }
    }

    fn trace_manhattan(&mut self, u: u32, v: u32) {
        if u == v {
            return;
        }

        let boundary_node = (self.parents.len() - 1) as u32;
        if u == boundary_node {
            self.emit_linear(v, u32::MAX);
            return;
        }
        if v == boundary_node {
            self.emit_linear(u, u32::MAX);
            return;
        }

        let (ux, uy, uz) = self.get_coord(u);
        let (vx, vy, vz) = self.get_coord(v);

        let dx = ux.abs_diff(vx);
        let dy = uy.abs_diff(vy);
        let dz = uz.abs_diff(vz);

        let mut curr_idx = u as usize;

        if dx > 0 {
            let stride = self.graph.stride_x;
            let step = if ux < vx {
                stride as isize
            } else {
                -(stride as isize)
            };
            for _ in 0..dx {
                let next_idx = (curr_idx as isize + step) as usize;
                self.emit_linear(curr_idx as u32, next_idx as u32);
                curr_idx = next_idx;
            }
        }

        if dy > 0 {
            let stride = self.stride_y;
            let step = if uy < vy {
                stride as isize
            } else {
                -(stride as isize)
            };
            for _ in 0..dy {
                let next_idx = (curr_idx as isize + step) as usize;
                self.emit_linear(curr_idx as u32, next_idx as u32);
                curr_idx = next_idx;
            }
        }

        if dz > 0 {
            let stride = self.graph.stride_z;
            let step = if uz < vz {
                stride as isize
            } else {
                -(stride as isize)
            };
            for _ in 0..dz {
                let next_idx = (curr_idx as isize + step) as usize;
                self.emit_linear(curr_idx as u32, next_idx as u32);
                curr_idx = next_idx;
            }
        }
    }

    fn emit_linear(&mut self, u: u32, v: u32) {
        if v == u32::MAX {
            let blk_idx = (u as usize) / 64;
            let bit_idx = (u as usize) % 64;

            let mask_idx = blk_idx >> 6;
            let mask_bit = blk_idx & 63;
            let m_ptr = unsafe { self.boundary_dirty_mask.get_unchecked_mut(mask_idx) };
            if (*m_ptr & (1 << mask_bit)) == 0 {
                *m_ptr |= 1 << mask_bit;
                unsafe {
                    *self
                        .boundary_dirty_list
                        .get_unchecked_mut(self.boundary_dirty_count) = blk_idx as u32;
                }
                self.boundary_dirty_count += 1;
            }
            let word_ptr = unsafe { self.boundary_bitmap.get_unchecked_mut(blk_idx) };
            *word_ptr ^= 1 << bit_idx;
            return;
        }

        let (u, v) = if u < v { (u, v) } else { (v, u) };
        let diff = v - u;

        let dir = if diff == 1 {
            0
        } else if diff == self.stride_y as u32 {
            1
        } else if diff == self.graph.stride_z as u32 {
            2
        } else {
            return;
        };

        let idx = (u as usize) * 3 + dir;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;

        let mask_idx = word_idx >> 6;
        let mask_bit = word_idx & 63;
        let m_ptr = unsafe { self.edge_dirty_mask.get_unchecked_mut(mask_idx) };
        if (*m_ptr & (1 << mask_bit)) == 0 {
            *m_ptr |= 1 << mask_bit;
            unsafe {
                *self
                    .edge_dirty_list
                    .get_unchecked_mut(self.edge_dirty_count) = word_idx as u32;
            }
            self.edge_dirty_count += 1;
        }
        let word_ptr = unsafe { self.edge_bitmap.get_unchecked_mut(word_idx) };
        *word_ptr ^= 1 << bit_idx;
    }

    fn get_coord(&self, u: u32) -> (usize, usize, usize) {
        let u = u as usize;
        if self.graph.depth > 1 {
            let z = u >> self.graph.shift_z;
            let rem = u & (self.graph.stride_z - 1);
            let y = rem >> self.graph.shift_y;
            let x = rem & (self.stride_y - 1);
            (x, y, z)
        } else {
            let y = u >> self.graph.shift_y;
            let x = u & (self.stride_y - 1);
            (x, y, 0)
        }
    }
}

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    #[inline(always)]
    fn trace_bitmask_bfs_impl(&mut self, start_node: u32, visited: &mut [u64]) {
        // Stack-based BFS for small grids (<= 4096 nodes + boundary sentinel).
        let visited_len = self.blocks_state.len().min(visited.len());

        // Destructure self to avoid borrow conflicts and overhead
        let bfs_pred = &mut *self.bfs_pred;
        let bfs_queue = &mut *self.bfs_queue;
        let edge_bitmap = &mut *self.edge_bitmap;
        let edge_dirty_list = &mut *self.edge_dirty_list;
        let edge_dirty_count = &mut self.edge_dirty_count;
        let boundary_bitmap = &mut *self.boundary_bitmap;
        let boundary_dirty_list = &mut *self.boundary_dirty_list;
        let boundary_dirty_count = &mut self.boundary_dirty_count;

        let blocks_state = &*self.blocks_state;

        let stride_y = self.stride_y as u32;
        let stride_z = self.graph.stride_z as u32;
        let shift_y = self.graph.shift_y;
        let shift_z = self.graph.shift_z;
        let mask_y = self.stride_y - 1;
        let mask_z = self.graph.stride_z - 1;

        let width = self.width;
        let height = self.height;
        let depth = self.graph.depth;
        let is_3d = depth > 1;

        let mut head = 0;
        let mut tail = 0;

        let start_blk = (start_node as usize) / 64;
        let start_bit = (start_node as usize) % 64;

        if start_blk < visited.len() {
            visited[start_blk] |= 1 << start_bit;
        }

        bfs_queue[tail] = start_node as u16;
        tail += 1;

        let mut boundary_hit = u32::MAX;

        let mut emit_linear = |u: u32, v: u32| {
            if v == u32::MAX {
                let blk_idx = (u as usize) / 64;
                let bit_idx = (u as usize) % 64;

                let word_ptr = unsafe { boundary_bitmap.get_unchecked_mut(blk_idx) };
                if *word_ptr == 0 {
                    unsafe {
                        *boundary_dirty_list.get_unchecked_mut(*boundary_dirty_count) =
                            blk_idx as u32;
                    }
                    *boundary_dirty_count += 1;
                }
                *word_ptr ^= 1 << bit_idx;
                return;
            }

            let (u, v) = if u < v { (u, v) } else { (v, u) };
            let diff = v - u;

            let dir = if diff == 1 {
                0
            } else if diff == stride_y {
                1
            } else if diff == stride_z {
                2
            } else {
                return;
            };

            let idx = (u as usize) * 3 + dir;
            let word_idx = idx / 64;
            let bit_idx = idx % 64;

            let word_ptr = unsafe { edge_bitmap.get_unchecked_mut(word_idx) };
            if *word_ptr == 0 {
                unsafe {
                    *edge_dirty_list.get_unchecked_mut(*edge_dirty_count) = word_idx as u32;
                }
                *edge_dirty_count += 1;
            }
            *word_ptr ^= 1 << bit_idx;
        };

        // Specialized Fast Path for 32x32 Grid
        if STRIDE_Y == 32 && !is_3d {
            while head != tail {
                let u = bfs_queue[head] as u32;
                head += 1;

                // Fast boundary check
                let x = u & 31;
                let y = u >> 5;
                if x == 0 || x == (width as u32 - 1) || y == 0 || y == (height as u32 - 1) {
                    boundary_hit = u;
                    break;
                }

                // Left (-1)
                let n = u - 1;
                let n_blk = (n as usize) >> 6;
                let n_bit = (n as usize) & 63;
                if n_blk < 17 {
                    let n_occ = unsafe { blocks_state.get_unchecked(n_blk).occupied };
                    if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                        visited[n_blk] |= 1 << n_bit;
                        bfs_pred[n as usize] = u as u16;
                        bfs_queue[tail] = n as u16;
                        tail += 1;
                    }
                }

                // Right (+1)
                let n = u + 1;
                let n_blk = (n as usize) >> 6;
                let n_bit = (n as usize) & 63;
                if n_blk < 17 {
                    let n_occ = unsafe { blocks_state.get_unchecked(n_blk).occupied };
                    if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                        visited[n_blk] |= 1 << n_bit;
                        bfs_pred[n as usize] = u as u16;
                        bfs_queue[tail] = n as u16;
                        tail += 1;
                    }
                }

                // Up (-32)
                let n = u - 32;
                let n_blk = (n as usize) >> 6;
                let n_bit = (n as usize) & 63;
                if n_blk < 17 {
                    let n_occ = unsafe { blocks_state.get_unchecked(n_blk).occupied };
                    if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                        visited[n_blk] |= 1 << n_bit;
                        bfs_pred[n as usize] = u as u16;
                        bfs_queue[tail] = n as u16;
                        tail += 1;
                    }
                }

                // Down (+32)
                let n = u + 32;
                let n_blk = (n as usize) >> 6;
                let n_bit = (n as usize) & 63;
                if n_blk < 17 {
                    let n_occ = unsafe { blocks_state.get_unchecked(n_blk).occupied };
                    if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                        visited[n_blk] |= 1 << n_bit;
                        bfs_pred[n as usize] = u as u16;
                        bfs_queue[tail] = n as u16;
                        tail += 1;
                    }
                }
            }
        } else {
            // Generic Path
            while head != tail {
                let u = bfs_queue[head] as u32;
                head += 1;

                // Inline get_coord
                let (x, y, z) = if is_3d {
                    let z_coord = (u as usize) >> shift_z;
                    let rem = (u as usize) & mask_z;
                    let y_coord = rem >> shift_y;
                    let x_coord = rem & mask_y;
                    (x_coord, y_coord, z_coord)
                } else {
                    let y_coord = (u as usize) >> shift_y;
                    let x_coord = (u as usize) & mask_y;
                    (x_coord, y_coord, 0)
                };

                if x == 0
                    || x == width - 1
                    || y == 0
                    || y == height - 1
                    || (is_3d && (z == 0 || z == depth - 1))
                {
                    boundary_hit = u;
                    break;
                }

                // Left
                if x > 0 {
                    let n = u - 1;
                    let n_blk = (n as usize) / 64;
                    let n_bit = (n as usize) % 64;
                    if n_blk < visited.len() {
                        let n_occ = blocks_state[n_blk].occupied;
                        if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                            visited[n_blk] |= 1 << n_bit;
                            bfs_pred[n as usize] = u as u16;
                            bfs_queue[tail] = n as u16;
                            tail += 1;
                        }
                    }
                }
                // Right
                if x < width - 1 {
                    let n = u + 1;
                    let n_blk = (n as usize) / 64;
                    let n_bit = (n as usize) % 64;
                    if n_blk < visited.len() {
                        let n_occ = blocks_state[n_blk].occupied;
                        if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                            visited[n_blk] |= 1 << n_bit;
                            bfs_pred[n as usize] = u as u16;
                            bfs_queue[tail] = n as u16;
                            tail += 1;
                        }
                    }
                }
                // Up
                if y > 0 {
                    let n = u - stride_y;
                    let n_blk = (n as usize) / 64;
                    let n_bit = (n as usize) % 64;
                    if n_blk < visited.len() {
                        let n_occ = blocks_state[n_blk].occupied;
                        if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                            visited[n_blk] |= 1 << n_bit;
                            bfs_pred[n as usize] = u as u16;
                            bfs_queue[tail] = n as u16;
                            tail += 1;
                        }
                    }
                }
                // Down
                if y < height - 1 {
                    let n = u + stride_y;
                    let n_blk = (n as usize) / 64;
                    let n_bit = (n as usize) % 64;
                    if n_blk < visited.len() {
                        let n_occ = blocks_state[n_blk].occupied;
                        if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                            visited[n_blk] |= 1 << n_bit;
                            bfs_pred[n as usize] = u as u16;
                            bfs_queue[tail] = n as u16;
                            tail += 1;
                        }
                    }
                }
                // Z-neighbors if 3D
                if is_3d {
                    if z > 0 {
                        let n = u - stride_z;
                        let n_blk = (n as usize) / 64;
                        let n_bit = (n as usize) % 64;
                        if n_blk < visited.len() {
                            let n_occ = blocks_state[n_blk].occupied;
                            if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                                visited[n_blk] |= 1 << n_bit;
                                bfs_pred[n as usize] = u as u16;
                                bfs_queue[tail] = n as u16;
                                tail += 1;
                            }
                        }
                    }
                    if z < depth - 1 {
                        let n = u + stride_z;
                        let n_blk = (n as usize) / 64;
                        let n_bit = (n as usize) % 64;
                        if n_blk < visited.len() {
                            let n_occ = blocks_state[n_blk].occupied;
                            if (n_occ & (1 << n_bit)) != 0 && (visited[n_blk] & (1 << n_bit)) == 0 {
                                visited[n_blk] |= 1 << n_bit;
                                bfs_pred[n as usize] = u as u16;
                                bfs_queue[tail] = n as u16;
                                tail += 1;
                            }
                        }
                    }
                }
            }
        }

        if boundary_hit != u32::MAX {
            let mut curr = boundary_hit;
            emit_linear(curr, u32::MAX);

            while curr != start_node {
                let p = bfs_pred[curr as usize];
                if p == u16::MAX {
                    break;
                }
                emit_linear(p as u32, curr);
                curr = p as u32;
            }
        }

        // Clear all defects in this component from defect_mask to prevent re-processing
        #[allow(clippy::needless_range_loop)]
        for i in 0..visited_len {
            if visited[i] != 0 && let Some(dm) = self.defect_mask.get_mut(i) {
                *dm &= !visited[i];
            }
        }
    }
}

#[inline(always)]
fn try_queue(
    next: usize,
    curr: usize,
    mask: u64,
    visited: &mut u64,
    queue: &mut u64,
    pred: &mut [u8; 64],
) {
    if (mask & (1 << next)) != 0 && (*visited & (1 << next)) == 0 {
        *visited |= 1 << next;
        *queue |= 1 << next;
        pred[next] = curr as u8;
    }
}



#[cfg(kani)]

mod kani_proofs;


