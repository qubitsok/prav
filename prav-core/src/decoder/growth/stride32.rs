use crate::decoder::state::DecodingState;
use crate::intrinsics::{prefetch_l1, tzcnt};
use crate::topology::Topology;

#[inline(always)]
fn saturate_row(b: u32, m: u32) -> u32 {
    let mut l = b;
    l |= (l << 1) & m;
    l |= (l << 2) & m;
    l |= (l << 4) & m;
    l |= (l << 8) & m;
    l |= (l << 16) & m;

    let mut r = b;
    r |= (r >> 1) & m;
    r |= (r >> 2) & m;
    r |= (r >> 4) & m;
    r |= (r >> 8) & m;
    r |= (r >> 16) & m;

    l | r
}

#[inline(always)]
fn spread_syndrome_2x32(boundary: u64, mask: u64) -> u64 {
    let mut b0 = boundary as u32;
    let mut b1 = (boundary >> 32) as u32;

    let m0 = mask as u32;
    let m1 = (mask >> 32) as u32;

    // Pass 1
    b0 = saturate_row(b0, m0);
    b1 = saturate_row(b1, m1);

    let v_down = b0 & m1;
    let v_up = b1 & m0;
    b1 |= v_down;
    b0 |= v_up;

    // Pass 2
    b0 = saturate_row(b0, m0);
    b1 = saturate_row(b1, m1);

    let v_down = b0 & m1;
    let v_up = b1 & m0;
    b1 |= v_down;
    b0 |= v_up;

    (b0 as u64) | ((b1 as u64) << 32)
}

#[inline(always)]
unsafe fn mark_block_dirty_slice(blk_idx: usize, block_dirty_mask: &mut [u64]) {
    let mask_idx = blk_idx >> 6;
    let mask_bit = blk_idx & 63;
    if mask_idx < block_dirty_mask.len() {
        *block_dirty_mask.get_unchecked_mut(mask_idx) |= 1 << mask_bit;
    }
}

// O(1) fast path for find - at p=0.001, ~95% of nodes are self-rooted
#[inline(always)]
unsafe fn find_in_slice(parents: &mut [u32], i: u32, block_dirty_mask: &mut [u64]) -> u32 {
    let p = *parents.get_unchecked(i as usize);
    if p == i {
        return i; // Fast path: self-rooted (most common case)
    }
    find_in_slice_slow(parents, i, p, block_dirty_mask)
}

// Cold path: actual path compression
#[inline(never)]
#[cold]
unsafe fn find_in_slice_slow(
    parents: &mut [u32],
    mut i: u32,
    mut p: u32,
    block_dirty_mask: &mut [u64],
) -> u32 {
    loop {
        let grandparent = *parents.get_unchecked(p as usize);
        if p == grandparent {
            return p; // Found root
        }
        // Path halving: point i to grandparent
        *parents.get_unchecked_mut(i as usize) = grandparent;
        mark_block_dirty_slice(i as usize >> 6, block_dirty_mask);
        i = grandparent;
        p = *parents.get_unchecked(i as usize);
    }
}

#[inline(always)]
unsafe fn union_roots_in_slice(
    parents: &mut [u32],
    root_u: u32,
    root_v: u32,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    block_dirty_mask: &mut [u64],
    block_offset: usize,
) -> bool {
    if root_u == root_v {
        return false;
    }

    if root_u < root_v {
        // u joins v - only invalidate u's block cache
        let blk_u = (root_u as usize) >> 6;
        // Check if blk_u is within our local range
        if blk_u >= block_offset && blk_u < block_offset + blocks_state.len() {
            blocks_state.get_unchecked_mut(blk_u - block_offset).root = u32::MAX;
        }
        mark_block_dirty_slice(blk_u, block_dirty_mask);
        *parents.get_unchecked_mut(root_u as usize) = root_v;
    } else {
        // v joins u - only invalidate v's block cache
        let blk_v = (root_v as usize) >> 6;
        if blk_v >= block_offset && blk_v < block_offset + blocks_state.len() {
            blocks_state.get_unchecked_mut(blk_v - block_offset).root = u32::MAX;
        }
        mark_block_dirty_slice(blk_v, block_dirty_mask);
        *parents.get_unchecked_mut(root_v as usize) = root_u;
    }
    true
}

#[inline(always)]
unsafe fn union_in_slice(
    parents: &mut [u32],
    u: u32,
    v: u32,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    block_dirty_mask: &mut [u64],
    block_offset: usize,
) -> bool {
    let root_u = find_in_slice(parents, u, block_dirty_mask);
    let root_v = find_in_slice(parents, v, block_dirty_mask);
    union_roots_in_slice(parents, root_u, root_v, blocks_state, block_dirty_mask, block_offset)
}

#[inline(always)]
unsafe fn push_next_slice(blk_idx: usize, queued_mask: &mut [u64], _is_small_grid: bool) {
    let mask_idx = blk_idx >> 6;
    let mask_bit = blk_idx & 63;
    if mask_idx < queued_mask.len() {
        *queued_mask.get_unchecked_mut(mask_idx) |= 1 << mask_bit;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn fast_grow_block_slice<const SILENT: bool>(
    blk_idx: usize,
    grow_mask: u64,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    _defect_mask: &mut [u64],
    block_dirty_mask: &mut [u64],
    queued_mask: &mut [u64],
    is_small_grid: bool,
    block_offset: usize,
) {
    if grow_mask == 0 {
        return;
    }

    let block = blocks_state.get_unchecked_mut(blk_idx);
    block.occupied |= grow_mask;
    block.boundary |= grow_mask;
    mark_block_dirty_slice(blk_idx + block_offset, block_dirty_mask);

    if !SILENT {
        push_next_slice(blk_idx + block_offset, queued_mask, is_small_grid);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn merge_shifted_portable_slice(
    parents: &mut [u32],
    mut mask: u64,
    base_src: usize,
    shift: isize,
    base_target: usize,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    block_dirty_mask: &mut [u64],
    block_offset: usize,
) -> bool {
    let mut expanded = false;
    let parents_ptr = parents.as_mut_ptr();

    while mask != 0 {
        let start_bit = tzcnt(mask) as usize;
        let shifted_mask = mask >> start_bit;
        let run_len = tzcnt(!shifted_mask) as usize;

        if run_len == 64 {
            mask = 0;
        } else {
            let clear_mask = !(((1u64 << run_len) - 1) << start_bit);
            mask &= clear_mask;
        }

        let u_start = (base_src as isize + start_bit as isize + shift) as usize;
        let v_start = base_target + start_bit;

        for k in 0..run_len {
            let u = (u_start + k) as u32;
            let v = (v_start + k) as u32;

            let pu = *parents_ptr.add(u as usize);
            let pv = *parents_ptr.add(v as usize);

            if pu == pv {
                continue;
            }

            if pu == u && pv == v {
                if u != v {
                    // Both self-rooted: direct union
                    if u < v {
                        // u joins v
                        let blk_u = (u as usize) >> 6;
                        if blk_u >= block_offset && blk_u < block_offset + blocks_state.len() {
                            blocks_state.get_unchecked_mut(blk_u - block_offset).root = u32::MAX;
                        }
                        *parents_ptr.add(u as usize) = v;
                        mark_block_dirty_slice(blk_u, block_dirty_mask);
                    } else {
                        // v joins u
                        let blk_v = (v as usize) >> 6;
                        if blk_v >= block_offset && blk_v < block_offset + blocks_state.len() {
                            blocks_state.get_unchecked_mut(blk_v - block_offset).root = u32::MAX;
                        }
                        *parents_ptr.add(v as usize) = u;
                        mark_block_dirty_slice(blk_v, block_dirty_mask);
                    }
                    expanded = true;
                }
            } else if union_in_slice(parents, u, v, blocks_state, block_dirty_mask, block_offset) {
                expanded = true;
            }
        }
    }
    expanded
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn union_mask_to_boundary_slice(
    parents: &mut [u32],
    mut mask: u64,
    base_global: usize,
    offset: isize,
    boundary_node: u32,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    block_dirty_mask: &mut [u64],
    block_offset: usize,
) -> bool {
    let mut expanded = false;
    while mask != 0 {
        let start_bit = tzcnt(mask) as usize;
        let shifted_mask = mask >> start_bit;
        let run_len = tzcnt(!shifted_mask) as usize;

        if run_len == 64 {
            mask = 0;
        } else {
            let clear_mask = !(((1u64 << run_len) - 1) << start_bit);
            mask &= clear_mask;
        }

        let u_start = (base_global as isize + offset + start_bit as isize) as usize;
        for k in 0..run_len {
            let u = (u_start + k) as u32;
            let root_u = find_in_slice(parents, u, block_dirty_mask);
            if root_u == boundary_node {
                continue;
            }
            if union_roots_in_slice(
                parents,
                root_u,
                boundary_node,
                blocks_state,
                block_dirty_mask,
                block_offset,
            ) {
                expanded = true;
            }
        }
    }
    expanded
}

#[inline(always)]
unsafe fn write_parents_mono_32_slice(
    parents: &mut [u32],
    mut mask: u64,
    base_target: usize,
    root_r: u32,
    block_dirty_mask: &mut [u64],
    _block_offset: usize,
) {
    let parents_ptr = parents.as_mut_ptr();

    if mask != 0 {
        mark_block_dirty_slice(base_target >> 6, block_dirty_mask);
    }

    while mask != 0 {
        let start_bit = tzcnt(mask) as usize;
        let shifted_mask = mask >> start_bit;
        let run_len = tzcnt(!shifted_mask) as usize;

        if run_len == 64 {
            mask = 0;
        } else {
            let clear_mask = !(((1u64 << run_len) - 1) << start_bit);
            mask &= clear_mask;
        }

        let ptr = parents_ptr.add(base_target + start_bit);
        for k in 0..run_len {
            *ptr.add(k) = root_r;
        }
    }
}

#[inline(always)]
unsafe fn merge_mono_32_slice(
    parents: &mut [u32],
    mut mask: u64,
    base_target: usize,
    root_r: u32,
    blocks_state: &mut [crate::decoder::state::BlockStateHot],
    block_dirty_mask: &mut [u64],
    block_offset: usize,
) -> bool {
    let mut expanded = false;
    while mask != 0 {
        let bit = tzcnt(mask) as usize;
        mask &= mask - 1;
        let target = (base_target + bit) as u32;

        if *parents.get_unchecked(target as usize) == root_r {
            continue;
        }

        if union_in_slice(parents, target, root_r, blocks_state, block_dirty_mask, block_offset) {
            expanded = true;
        }
    }
    expanded
}

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `blk_idx` is within bounds of `blocks_state`
    /// - `parents`, `blocks_state`, `defect_mask`, `block_dirty_mask`, `queued_mask` are valid
    /// - `block_offset + blk_idx` corresponds to valid parent array indices
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn process_block_small_stride_32<const SILENT: bool>(
        blk_idx: usize,
        parents: &mut [u32],
        blocks_state: &mut [crate::decoder::state::BlockStateHot],
        defect_mask: &mut [u64],
        block_dirty_mask: &mut [u64],
        queued_mask: &mut [u64],
        is_small_grid: bool,
        block_offset: usize,
    ) -> bool {
        let mut expanded = false;
        let base_global = (blk_idx + block_offset) * 64;

        let p_ptr = parents.as_ptr().add(base_global) as *const u8;
        prefetch_l1(p_ptr);
        prefetch_l1(p_ptr.add(64));
        prefetch_l1(p_ptr.add(128));
        prefetch_l1(p_ptr.add(192));

        // Speculative prefetch for neighbor parent arrays
        if blk_idx > 0 {
            let up_ptr = parents.as_ptr().add((blk_idx + block_offset - 1) * 64) as *const u8;
            prefetch_l1(up_ptr);
            prefetch_l1(up_ptr.add(128));
        }
        if blk_idx + 1 < blocks_state.len() {
            let down_ptr = parents.as_ptr().add((blk_idx + block_offset + 1) * 64) as *const u8;
            prefetch_l1(down_ptr);
            prefetch_l1(down_ptr.add(128));
        }

        let boundary_node = (parents.len() - 1) as u32;

        let mut boundary_val = blocks_state.get_unchecked(blk_idx).boundary;

        if boundary_val == 0 {
            return false;
        }

        let occupied_val = blocks_state.get_unchecked(blk_idx).occupied;
        let initial_boundary = boundary_val;

        let valid_mask = blocks_state.get_unchecked(blk_idx).valid_mask;
        let erasure_mask_val = !blocks_state.get_unchecked(blk_idx).erasure_mask;
        let mask = valid_mask & erasure_mask_val;

        let mut spread_boundary = spread_syndrome_2x32(boundary_val, mask);
        let up = (spread_boundary << 32) & mask;
        let down = (spread_boundary >> 32) & mask;
        spread_boundary |= up | down;

        // --- O(1) Monochromatic Check using cached root ---
        // At p=0.001, ~95% of nodes are self-rooted, and the cache
        // is invalidated on any union, so trusting it avoids O(popcount) find() calls
        let mut is_monochromatic;
        let root_r;
        {
            let cached_root = blocks_state.get_unchecked(blk_idx).root;
            if cached_root != u32::MAX {
                // Cache hit - validate with single O(1) check
                let p = *parents.get_unchecked(cached_root as usize);
                root_r = if p == cached_root {
                    cached_root // Root unchanged (most common)
                } else {
                    // Root was updated by a union; find the current root
                    find_in_slice(parents, cached_root, block_dirty_mask)
                };
                is_monochromatic = true;
            } else {
                // Cache miss - must verify monochromatic status
                let first_bit = tzcnt(boundary_val) as usize;
                root_r = find_in_slice(parents, (base_global + first_bit) as u32, block_dirty_mask);
                is_monochromatic = true;

                // Check remaining boundary bits for different roots
                let mut temp = boundary_val & !(1u64 << first_bit);
                while temp != 0 {
                    let bit = tzcnt(temp) as usize;
                    temp &= temp - 1;

                    let node = (base_global + bit) as u32;
                    // Fast path: direct parent check
                    if *parents.get_unchecked(node as usize) == root_r {
                        continue;
                    }
                    // Slow path: full find
                    if find_in_slice(parents, node, block_dirty_mask) != root_r {
                        is_monochromatic = false;
                        break;
                    }
                }

                // If monochromatic, cache the root for next iteration
                if is_monochromatic {
                    blocks_state.get_unchecked_mut(blk_idx).root = root_r;
                }
            }
        }

        if is_monochromatic {
            // --- Fast Path: Monochromatic ---

            let newly_occupied = spread_boundary & !occupied_val;
            if newly_occupied != 0 {
                write_parents_mono_32_slice(
                    parents,
                    newly_occupied,
                    base_global,
                    root_r,
                    block_dirty_mask,
                    block_offset,
                );
            }

            let _boundary_down_internal =
                (spread_boundary << 32) & !valid_mask & 0xFFFFFFFF00000000;
            // Up Neighbor (Block - 1)
            if blk_idx > 0 {
                let blk_up = blk_idx - 1;
                let valid_up = blocks_state.get_unchecked(blk_up).valid_mask;

                let spread_to_up = (spread_boundary & 0xFFFFFFFF) << 32;

                let hits = (spread_boundary & 0xFFFFFFFF) & (!valid_up >> 32);
                if hits != 0
                    && union_roots_in_slice(
                        parents,
                        root_r,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }

                if valid_up != 0 {
                    let occupied_up = blocks_state.get_unchecked(blk_up).occupied;
                    let erasure_up = !blocks_state.get_unchecked(blk_up).erasure_mask;

                    let grow_up = spread_to_up & !occupied_up & valid_up & erasure_up;
                    if grow_up != 0 {
                        fast_grow_block_slice::<SILENT>(
                            blk_up,
                            grow_up,
                            blocks_state,
                            defect_mask,
                            block_dirty_mask,
                            queued_mask,
                            is_small_grid,
                            block_offset,
                        );
                        write_parents_mono_32_slice(
                            parents,
                            grow_up,
                            (blk_up + block_offset) * 64,
                            root_r,
                            block_dirty_mask,
                            block_offset,
                        );
                        expanded = true;
                    }

                    let merge_up = spread_to_up & occupied_up;
                    if merge_up != 0
                        && merge_mono_32_slice(
                            parents,
                            merge_up,
                            (blk_up + block_offset) * 64,
                            root_r,
                            blocks_state,
                            block_dirty_mask,
                            block_offset,
                        )
                    {
                        expanded = true;
                    }
                }
            } else if (spread_boundary & 0xFFFFFFFF) != 0
                && union_roots_in_slice(
                    parents,
                    root_r,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            // Down Neighbor (Block + 1)
            if blk_idx + 1 < blocks_state.len() {
                let blk_down = blk_idx + 1;
                let valid_down = blocks_state.get_unchecked(blk_down).valid_mask;
                let spread_to_down = spread_boundary >> 32;

                let hits = spread_to_down & !valid_down;
                if hits != 0
                    && union_roots_in_slice(
                        parents,
                        root_r,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }

                if valid_down != 0 {
                    let occupied_down = blocks_state.get_unchecked(blk_down).occupied;
                    let erasure_down = !blocks_state.get_unchecked(blk_down).erasure_mask;

                    let grow_down = spread_to_down & !occupied_down & valid_down & erasure_down;
                    if grow_down != 0 {
                        fast_grow_block_slice::<SILENT>(
                            blk_down,
                            grow_down,
                            blocks_state,
                            defect_mask,
                            block_dirty_mask,
                            queued_mask,
                            is_small_grid,
                            block_offset,
                        );
                        write_parents_mono_32_slice(
                            parents,
                            grow_down,
                            (blk_down + block_offset) * 64,
                            root_r,
                            block_dirty_mask,
                            block_offset,
                        );
                        expanded = true;
                    }

                    let merge_down = spread_to_down & occupied_down;
                    if merge_down != 0
                        && merge_mono_32_slice(
                            parents,
                            merge_down,
                            (blk_down + block_offset) * 64,
                            root_r,
                            blocks_state,
                            block_dirty_mask,
                            block_offset,
                        )
                    {
                        expanded = true;
                    }
                }
            } else if (spread_boundary & 0xFFFFFFFF00000000) != 0
                && union_roots_in_slice(
                    parents,
                    root_r,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            if (spread_boundary & 0x0000000100000001) != 0
                && union_roots_in_slice(
                    parents,
                    root_r,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            if (spread_boundary & 0x8000000080000000) != 0
                && union_roots_in_slice(
                    parents,
                    root_r,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }
        } else {
            // --- Slow Path: Polychromatic ---

            let vertical_pairs = spread_boundary & (spread_boundary >> 32) & 0xFFFFFFFF;
            if vertical_pairs != 0 {
                let mut temp = vertical_pairs;
                while temp != 0 {
                    let bit = tzcnt(temp) as usize;
                    temp &= temp - 1;
                    let u = (base_global + bit) as u32;
                    let v = (base_global + bit + 32) as u32;
                    if union_in_slice(parents, u, v, blocks_state, block_dirty_mask, block_offset) {
                        expanded = true;
                    }
                }
            }

            let horizontal_pairs = spread_boundary & (spread_boundary >> 1) & 0x7FFFFFFF7FFFFFFF;
            if horizontal_pairs != 0
                && merge_shifted_portable_slice(
                    parents,
                    horizontal_pairs,
                    base_global,
                    1,
                    base_global,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            let boundary_down_internal = (spread_boundary << 32) & !valid_mask & 0xFFFFFFFF00000000;
            if boundary_down_internal != 0
                && union_mask_to_boundary_slice(
                    parents,
                    boundary_down_internal,
                    base_global,
                    -32,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            let boundary_up_internal = (spread_boundary >> 32) & !valid_mask & 0x00000000FFFFFFFF;
            if boundary_up_internal != 0
                && union_mask_to_boundary_slice(
                    parents,
                    boundary_up_internal,
                    base_global,
                    32,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            // Inter-Block Connections

            // Up Neighbor (Block - 1)
            if blk_idx > 0 {
                let blk_up = blk_idx - 1;
                let valid_up = blocks_state.get_unchecked(blk_up).valid_mask;

                let spread_to_up = (spread_boundary & 0xFFFFFFFF) << 32;

                let hits = (spread_boundary & 0xFFFFFFFF) & (!valid_up >> 32);
                if hits != 0
                    && union_mask_to_boundary_slice(
                        parents,
                        hits,
                        base_global,
                        0,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }

                if valid_up != 0 {
                    let occupied_up = blocks_state.get_unchecked(blk_up).occupied;
                    let erasure_up = !blocks_state.get_unchecked(blk_up).erasure_mask;

                    let grow_up = spread_to_up & !occupied_up & valid_up & erasure_up;
                    if grow_up != 0 {
                        fast_grow_block_slice::<SILENT>(
                            blk_up,
                            grow_up,
                            blocks_state,
                            defect_mask,
                            block_dirty_mask,
                            queued_mask,
                            is_small_grid,
                            block_offset,
                        );
                        expanded = true;
                    }

                    let merge_up = spread_to_up & occupied_up;
                    if merge_up != 0
                        && merge_shifted_portable_slice(
                            parents,
                            merge_up,
                            base_global,
                            -32,
                            (blk_up + block_offset) * 64,
                            blocks_state,
                            block_dirty_mask,
                            block_offset,
                        )
                    {
                        expanded = true;
                    }
                }
            } else {
                let hits = spread_boundary & 0xFFFFFFFF;
                if hits != 0
                    && union_mask_to_boundary_slice(
                        parents,
                        hits,
                        base_global,
                        0,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }
            }

            // Down Neighbor (Block + 1)
            if blk_idx + 1 < blocks_state.len() {
                let blk_down = blk_idx + 1;
                let valid_down = blocks_state.get_unchecked(blk_down).valid_mask;
                let spread_to_down = spread_boundary >> 32;

                let hits = spread_to_down & !valid_down;
                if hits != 0
                    && union_mask_to_boundary_slice(
                        parents,
                        hits << 32,
                        base_global,
                        0,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }

                if valid_down != 0 {
                    let occupied_down = blocks_state.get_unchecked(blk_down).occupied;
                    let erasure_down = !blocks_state.get_unchecked(blk_down).erasure_mask;

                    let grow_down = spread_to_down & !occupied_down & valid_down & erasure_down;
                    if grow_down != 0 {
                        fast_grow_block_slice::<SILENT>(
                            blk_down,
                            grow_down,
                            blocks_state,
                            defect_mask,
                            block_dirty_mask,
                            queued_mask,
                            is_small_grid,
                            block_offset,
                        );
                        expanded = true;
                    }

                    let merge_down = spread_to_down & occupied_down;
                    if merge_down != 0
                        && merge_shifted_portable_slice(
                            parents,
                            merge_down,
                            base_global,
                            32,
                            (blk_down + block_offset) * 64,
                            blocks_state,
                            block_dirty_mask,
                            block_offset,
                        )
                    {
                        expanded = true;
                    }
                }
            } else {
                let hits = spread_boundary & 0xFFFFFFFF00000000;
                if hits != 0
                    && union_mask_to_boundary_slice(
                        parents,
                        hits,
                        base_global,
                        0,
                        boundary_node,
                        blocks_state,
                        block_dirty_mask,
                        block_offset,
                    )
                {
                    expanded = true;
                }
            }

            let left_edge_mask = 0x0000000100000001;
            let left_hits = spread_boundary & left_edge_mask;
            if left_hits != 0
                && union_mask_to_boundary_slice(
                    parents,
                    left_hits,
                    base_global,
                    0,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }

            let right_edge_mask = 0x8000000080000000;
            let right_hits = spread_boundary & right_edge_mask;
            if right_hits != 0
                && union_mask_to_boundary_slice(
                    parents,
                    right_hits,
                    base_global,
                    0,
                    boundary_node,
                    blocks_state,
                    block_dirty_mask,
                    block_offset,
                )
            {
                expanded = true;
            }
        }

        let new_occupied = occupied_val | spread_boundary;
        boundary_val &= !spread_boundary;

        if occupied_val != new_occupied || boundary_val != initial_boundary {
            let block = blocks_state.get_unchecked_mut(blk_idx);
            block.occupied = new_occupied;
            block.boundary = boundary_val;
            mark_block_dirty_slice(blk_idx + block_offset, block_dirty_mask);
            expanded = true;
            if !SILENT {
                push_next_slice(blk_idx + block_offset, queued_mask, is_small_grid);
            }
        }

        expanded
    }
}