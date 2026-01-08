use crate::decoder::state::DecodingState;
use crate::decoder::union_find::UnionFind;
use crate::topology::Topology;

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    pub(crate) unsafe fn merge_mono(
        &mut self,
        mut mask: u64,
        base_target: usize,
        root_source: u32,
    ) -> bool {
        if mask == 0 {
            return false;
        }

        let mut expanded = false;
        let mut roots = [0u32; 8]; // Stack-allocated buffer for unique roots
        let mut root_count = 0usize;

        // Phase 1: Gather unique roots from target nodes
        while mask != 0 && root_count < 8 {
            let bit = mask.trailing_zeros();
            mask &= mask - 1;
            let target_node = (base_target + bit as usize) as u32;

            let p = *self.parents.get_unchecked(target_node as usize);

            // Skip if already connected to root_source
            if p == root_source {
                continue;
            }

            // Fast path: node is its own root
            let target_root = if p == target_node {
                p
            } else {
                self.find(target_node)
            };

            // Skip if find() resolved to root_source
            if target_root == root_source {
                continue;
            }

            // Check if already collected (linear scan - small array)
            let mut found = false;
            for i in 0..root_count {
                if roots[i] == target_root {
                    found = true;
                    break;
                }
            }

            if !found {
                roots[root_count] = target_root;
                root_count += 1;
            }
        }

        // Phase 2: Bulk union unique roots with root_source
        for i in 0..root_count {
            if self.union_roots(roots[i], root_source) {
                expanded = true;
            }
        }

        // Phase 3: Handle overflow (remaining bits if buffer was full)
        while mask != 0 {
            let bit = mask.trailing_zeros();
            mask &= mask - 1;
            let target_node = (base_target + bit as usize) as u32;
            if self.union(target_node, root_source) {
                expanded = true;
            }
        }

        expanded
    }

    /// Optimized path for monochromatic blocks (>95% of blocks at p=0.001-0.06).
    ///
    /// Key optimization: When neighbor blocks share the same root (same cluster),
    /// we skip ALL union logic and use pure bitwise spreading.
    #[inline(always)]
    unsafe fn process_mono<const SILENT: bool>(
        &mut self,
        blk_idx: usize,
        mut canonical_root: u32,
    ) -> bool {
        let mut expanded = false;
        let base_global = blk_idx * 64;
        let boundary_node = (self.parents.len() - 1) as u32;

        // Prefetch current block parent array
        let p_ptr = self.parents.as_ptr().add(base_global) as *const u8;
        crate::intrinsics::prefetch_l1(p_ptr);
        crate::intrinsics::prefetch_l1(p_ptr.add(64));
        crate::intrinsics::prefetch_l1(p_ptr.add(128));
        crate::intrinsics::prefetch_l1(p_ptr.add(192));

        // Prefetch neighbor block states and parent arrays
        if blk_idx > 0 {
            let ptr = self.blocks_state.as_ptr().add(blk_idx - 1) as *const u8;
            crate::intrinsics::prefetch_l1(ptr);
            let p_ptr = self.parents.as_ptr().add((blk_idx - 1) * 64) as *const u8;
            crate::intrinsics::prefetch_l1(p_ptr);
            crate::intrinsics::prefetch_l1(p_ptr.add(128));
        }
        if blk_idx + 1 < self.blocks_state.len() {
            let ptr = self.blocks_state.as_ptr().add(blk_idx + 1) as *const u8;
            crate::intrinsics::prefetch_l1(ptr);
            let p_ptr = self.parents.as_ptr().add((blk_idx + 1) * 64) as *const u8;
            crate::intrinsics::prefetch_l1(p_ptr);
            crate::intrinsics::prefetch_l1(p_ptr.add(128));
        }

        // Load block state into local variables to minimize memory traffic
        // and allow single-writeback optimization.
        let (mut occupied, mut boundary, effective_mask, valid_mask, _flags, cached_root_in_memory) = {
            let b = self.blocks_state.get_unchecked(blk_idx);
            (
                b.occupied,
                b.boundary,
                b.effective_mask,
                b.valid_mask,
                b.flags,
                b.root,
            )
        };

        if boundary == 0 {
            return false;
        }

        let initial_occupied = occupied;
        let initial_boundary = boundary;

        // Validate cached root (O(1) check)
        let parent_of_root = *self.parents.get_unchecked(canonical_root as usize);
        if parent_of_root != canonical_root {
            // Root was updated by a union; find actual root
            canonical_root = self.find(canonical_root);
            // Intermediate write removed - will be handled by single writeback
        }

        // Calculate spread (horizontal)
        let mut spread_boundary = if STRIDE_Y == 32 {
            crate::intrinsics::spread_syndrome_masked(
                boundary,
                effective_mask,
                0x8000000080000000,
                0x0000000100000001,
            )
        } else {
            crate::intrinsics::spread_syndrome_masked(
                boundary,
                effective_mask,
                self.row_end_mask,
                self.row_start_mask,
            )
        };

        // Intra-block vertical spread (logarithmic)
        if STRIDE_Y < 64 {
            let mask = effective_mask;
            let mut current = spread_boundary;

            {
                let up = (current << STRIDE_Y) & mask;
                let down = (current >> STRIDE_Y) & mask;
                current |= up | down;
            }
            let shift_2 = STRIDE_Y * 2;
            if shift_2 < 64 {
                let up = (current << shift_2) & mask;
                let down = (current >> shift_2) & mask;
                current |= up | down;
            }
            let shift_4 = STRIDE_Y * 4;
            if shift_4 < 64 {
                let up = (current << shift_4) & mask;
                let down = (current >> shift_4) & mask;
                current |= up | down;
            }
            let shift_8 = STRIDE_Y * 8;
            if shift_8 < 64 {
                let up = (current << shift_8) & mask;
                let down = (current >> shift_8) & mask;
                current |= up | down;
            }
            let shift_16 = STRIDE_Y * 16;
            if shift_16 < 64 {
                let up = (current << shift_16) & mask;
                let down = (current >> shift_16) & mask;
                current |= up | down;
            }
            let shift_32 = STRIDE_Y * 32;
            if shift_32 < 64 {
                let up = (current << shift_32) & mask;
                let down = (current >> shift_32) & mask;
                current |= up | down;
            }

            spread_boundary = current;
        }

        // Direct parent assignment for new nodes (no union needed!)
        let newly_occupied = spread_boundary & !occupied;
        let mut temp_new = newly_occupied;
        while temp_new != 0 {
            let bit1 = temp_new.trailing_zeros();
            temp_new &= temp_new - 1;
            *self.parents.get_unchecked_mut(base_global + bit1 as usize) = canonical_root;

            if temp_new == 0 {
                break;
            }
            let bit2 = temp_new.trailing_zeros();
            temp_new &= temp_new - 1;
            *self.parents.get_unchecked_mut(base_global + bit2 as usize) = canonical_root;

            if temp_new == 0 {
                break;
            }
            let bit3 = temp_new.trailing_zeros();
            temp_new &= temp_new - 1;
            *self.parents.get_unchecked_mut(base_global + bit3 as usize) = canonical_root;

            if temp_new == 0 {
                break;
            }
            let bit4 = temp_new.trailing_zeros();
            temp_new &= temp_new - 1;
            *self.parents.get_unchecked_mut(base_global + bit4 as usize) = canonical_root;
        }

        // Internal boundary connections (holes)
        if STRIDE_Y < 64 {
            let internal_bottom_edge = spread_boundary & (!valid_mask >> STRIDE_Y);
            let internal_top_edge = spread_boundary & (!valid_mask << STRIDE_Y);
            let internal_hits = internal_bottom_edge | internal_top_edge;

            if internal_hits != 0 && self.union_roots(canonical_root, boundary_node) {
                expanded = true;
                if canonical_root < boundary_node {
                    canonical_root = boundary_node;
                }
            }
        }

        let new_occupied = occupied | spread_boundary;
        if new_occupied != occupied {
            occupied = new_occupied;
            expanded = true;
            if !SILENT {
                self.push_next(blk_idx);
            }
        }

        // ============== NEIGHBOR PROCESSING ==============
        // Up neighbor (blk_idx - 1)
        if blk_idx > 0 {
            let blk_up = blk_idx - 1;
            let (valid_up, occupied_up, effective_up, r_up_val) = {
                let b = self.blocks_state.get_unchecked(blk_up);
                (b.valid_mask, b.occupied, b.effective_mask, b.root)
            };
            let shift_amt = 64 - STRIDE_Y;
            let spread_to_up = spread_boundary << shift_amt;

            // Boundary connection check
            let boundary_connect_mask = spread_to_up & !valid_up;
            if boundary_connect_mask != 0 {
                // Only connect if Top boundary is enabled for this DecodingState
                // For internal tiles, this might be disabled.
                // However, blk_idx > 0 means we are INSIDE the tile connecting to previous block.
                // This is an INTERNAL edge, not a global boundary.
                // Global boundary is when we try to connect 'up' from block 0, or to invalid bits.
                // Wait, !valid_up means bits that are NOT valid in the up block.
                // If the Up Block has invalid bits, it means we hit a hole or edge.
                // If it's a hole, we connect to boundary (virtual).
                // If it's the edge of the grid, we connect to boundary.
                if self.union_roots(canonical_root, boundary_node) {
                    expanded = true;
                    if canonical_root < boundary_node {
                        canonical_root = boundary_node;
                    }
                }
            }

            if valid_up != 0 {
                let grow_up = spread_to_up & !occupied_up & effective_up;
                let merge_up = spread_to_up & occupied_up;

                // CRITICAL OPTIMIZATION: Check if neighbor has same root (same cluster)
                if r_up_val != u32::MAX {
                    // Fast path: check if cached roots match (common case at p=0.001)
                    if r_up_val == canonical_root {
                        // Same cluster - skip all union logic
                        if grow_up != 0 {
                            self.fast_grow_block::<SILENT>(blk_up, grow_up);
                            expanded = true;
                        }
                        // merge_up handled implicitly (self-loop)
                    } else {
                        // Different clusters OR need to verify roots
                        let neighbor_root = unsafe {
                            let p = *self.parents.get_unchecked(r_up_val as usize);
                            if p == r_up_val {
                                r_up_val
                            } else {
                                self.find(r_up_val)
                            }
                        };
                        if neighbor_root == canonical_root {
                            // Same cluster after root resolution
                            if grow_up != 0 {
                                self.fast_grow_block::<SILENT>(blk_up, grow_up);
                                expanded = true;
                            }
                        } else {
                            // Different clusters - perform union
                            if (merge_up != 0 || grow_up != 0)
                                && self.union_roots(canonical_root, neighbor_root)
                            {
                                expanded = true;
                                if canonical_root < neighbor_root {
                                    canonical_root = neighbor_root;
                                }
                            }
                            if grow_up != 0 {
                                self.fast_grow_block::<SILENT>(blk_up, grow_up);
                                expanded = true;
                            }
                        }
                    }
                } else {
                    // Neighbor is polychromatic - use merge_mono
                    if grow_up != 0 {
                        self.fast_grow_block::<SILENT>(blk_up, grow_up);
                        expanded = true;
                    }
                    if merge_up != 0 {
                        if self.merge_mono(merge_up, blk_up * 64, canonical_root) {
                            expanded = true;
                        }
                        canonical_root = self.find(canonical_root);
                    }
                }
            }
        } else {
            // Top block - check boundary
            if self.boundary_config.check_top {
                if STRIDE_Y < 64 {
                    let mask_low = (1u64 << STRIDE_Y) - 1;
                    let spread_boundary_masked = spread_boundary & mask_low;
                    if spread_boundary_masked != 0
                        && self.union_roots(canonical_root, boundary_node)
                    {
                        expanded = true;
                        if canonical_root < boundary_node {
                            canonical_root = boundary_node;
                        }
                    }
                } else if STRIDE_Y == 64 {
                    // Stride 64: All active nodes in block 0 are on top boundary
                    if self.union_roots(canonical_root, boundary_node) {
                        expanded = true;
                        if canonical_root < boundary_node {
                            canonical_root = boundary_node;
                        }
                    }
                }
            }
        }

        // Down neighbor (blk_idx + 1)
        if blk_idx + 1 < self.blocks_state.len() {
            let blk_down = blk_idx + 1;
            let (valid_down, occupied_down, effective_down, r_down_val) = {
                let b = self.blocks_state.get_unchecked(blk_down);
                (b.valid_mask, b.occupied, b.effective_mask, b.root)
            };
            let shift_amt = 64 - STRIDE_Y;
            let spread_to_down = spread_boundary >> shift_amt;

            // Boundary connection check
            let boundary_connect_mask = spread_to_down & !valid_down;
            if boundary_connect_mask != 0 {
                // Internal tile connection - always try to union if it's a hole.
                // But for tile boundaries, we rely on check_bottom.
                // Actually, if we are internal, valid_mask should be full?
                // If valid_mask has 0s, it's a hole.
                // Holes connect to global boundary.
                // If we are at the bottom of the tile (which is internal),
                // the `blk_idx` check ensures we are not at the very last block.
                // So this branch runs for internal blocks.
                // If an internal block connects to a hole (invalid bit in neighbor),
                // it should connect to boundary.
                if self.union_roots(canonical_root, boundary_node) {
                    expanded = true;
                    if canonical_root < boundary_node {
                        canonical_root = boundary_node;
                    }
                }
            }

            if valid_down != 0 {
                let grow_down = spread_to_down & !occupied_down & effective_down;
                let merge_down = spread_to_down & occupied_down;

                // CRITICAL OPTIMIZATION: Check if neighbor has same root
                if r_down_val != u32::MAX {
                    // Fast path: check if cached roots match
                    if r_down_val == canonical_root {
                        // Same cluster - just spread
                        if grow_down != 0 {
                            self.fast_grow_block::<SILENT>(blk_down, grow_down);
                            expanded = true;
                        }
                        // merge_down is a self-loop, skip entirely
                    } else {
                        // Different clusters OR need to verify roots
                        let neighbor_root = unsafe {
                            let p = *self.parents.get_unchecked(r_down_val as usize);
                            if p == r_down_val {
                                r_down_val
                            } else {
                                self.find(r_down_val)
                            }
                        };
                        if neighbor_root == canonical_root {
                            // Same cluster after root resolution
                            if grow_down != 0 {
                                self.fast_grow_block::<SILENT>(blk_down, grow_down);
                                expanded = true;
                            }
                        } else {
                            // Different clusters - perform union
                            if (merge_down != 0 || grow_down != 0)
                                && self.union_roots(canonical_root, neighbor_root)
                            {
                                expanded = true;
                                if canonical_root < neighbor_root {
                                    canonical_root = neighbor_root;
                                }
                            }
                            if grow_down != 0 {
                                self.fast_grow_block::<SILENT>(blk_down, grow_down);
                                expanded = true;
                            }
                        }
                    }
                } else {
                    // Neighbor is polychromatic - use merge_mono
                    if grow_down != 0 {
                        self.fast_grow_block::<SILENT>(blk_down, grow_down);
                        expanded = true;
                    }
                    if merge_down != 0 {
                        if self.merge_mono(merge_down, blk_down * 64, canonical_root) {
                            expanded = true;
                        }
                        canonical_root = self.find(canonical_root);
                    }
                }
            }
        } else {
            // Last block - bottom boundary
            if self.boundary_config.check_bottom {
                if STRIDE_Y < 64 {
                    let shift_amt = 64 - STRIDE_Y;
                    let spread_boundary_masked = spread_boundary >> shift_amt;
                    if spread_boundary_masked != 0
                        && self.union_roots(canonical_root, boundary_node)
                    {
                        expanded = true;
                        if canonical_root < boundary_node {
                            canonical_root = boundary_node;
                        }
                    }
                } else {
                    // Stride 64: All active nodes connect to boundary
                    if self.union_roots(canonical_root, boundary_node) {
                        expanded = true;
                        if canonical_root < boundary_node {
                            canonical_root = boundary_node;
                        }
                    }
                }
            }
        }

        // Horizontal edges
        let left_edge = spread_boundary & self.row_start_mask;
        if left_edge != 0
            && self.boundary_config.check_left
            && self.union_roots(canonical_root, boundary_node)
        {
            expanded = true;
            if canonical_root < boundary_node {
                canonical_root = boundary_node;
            }
        }

        let right_edge = spread_boundary & self.row_end_mask;
        if right_edge != 0
            && self.boundary_config.check_right
            && self.union_roots(canonical_root, boundary_node)
        {
            expanded = true;
            if canonical_root < boundary_node {
                canonical_root = boundary_node;
            }
        }

        // Clean up boundary
        boundary &= !spread_boundary;

        // Single writeback of all state changes
        if occupied != initial_occupied
            || boundary != initial_boundary
            || expanded
            || canonical_root != cached_root_in_memory
        {
            let block = self.blocks_state.get_unchecked_mut(blk_idx);
            block.occupied = occupied;
            block.boundary = boundary;
            block.root = canonical_root;
            self.mark_block_dirty(blk_idx);
        }

        expanded
    }

    /// Polychromatic path - handles blocks with multiple roots.
    /// This is the slow path, used when block.root == u32::MAX.
    #[inline(always)]
    unsafe fn process_poly<const SILENT: bool>(&mut self, blk_idx: usize) -> bool {
        let mut expanded = false;
        let base_global = self.parent_offset + blk_idx * 64;
        let boundary_node = (self.parents.len() - 1) as u32;

        // Prefetch current block parent array
        let p_ptr = self.parents.as_ptr().add(base_global) as *const u8;
        crate::intrinsics::prefetch_l1(p_ptr);
        crate::intrinsics::prefetch_l1(p_ptr.add(64));
        crate::intrinsics::prefetch_l1(p_ptr.add(128));
        crate::intrinsics::prefetch_l1(p_ptr.add(192));

        // Prefetch neighbor block states and parent arrays
        if blk_idx > 0 {
            let ptr = self.blocks_state.as_ptr().add(blk_idx - 1) as *const u8;
            crate::intrinsics::prefetch_l1(ptr);
            let p_ptr = self.parents.as_ptr().add((blk_idx - 1) * 64) as *const u8;
            crate::intrinsics::prefetch_l1(p_ptr);
            crate::intrinsics::prefetch_l1(p_ptr.add(128));
        }
        if blk_idx + 1 < self.blocks_state.len() {
            let ptr = self.blocks_state.as_ptr().add(blk_idx + 1) as *const u8;
            crate::intrinsics::prefetch_l1(ptr);
            let p_ptr = self.parents.as_ptr().add((blk_idx + 1) * 64) as *const u8;
            crate::intrinsics::prefetch_l1(p_ptr);
            crate::intrinsics::prefetch_l1(p_ptr.add(128));
        }

        let block_state_ptr = self.blocks_state.get_unchecked_mut(blk_idx);
        let _flags = block_state_ptr.flags;
        let mut boundary = block_state_ptr.boundary;

        if boundary == 0 {
            return false;
        }

        let mut occupied = block_state_ptr.occupied;
        let initial_occupied = occupied;
        let initial_boundary = boundary;
        let effective_mask = block_state_ptr.effective_mask;
        let valid_mask = block_state_ptr.valid_mask;

        // Calculate spread (horizontal)
        let mut spread_boundary = if STRIDE_Y == 32 {
            crate::intrinsics::spread_syndrome_masked(
                boundary,
                effective_mask,
                0x8000000080000000,
                0x0000000100000001,
            )
        } else {
            crate::intrinsics::spread_syndrome_masked(
                boundary,
                effective_mask,
                self.row_end_mask,
                self.row_start_mask,
            )
        };

        // Intra-block vertical spread (logarithmic)
        if STRIDE_Y < 64 {
            let mask = effective_mask;
            let mut current = spread_boundary;

            {
                let up = (current << STRIDE_Y) & mask;
                let down = (current >> STRIDE_Y) & mask;
                current |= up | down;
            }
            let shift_2 = STRIDE_Y * 2;
            if shift_2 < 64 {
                let up = (current << shift_2) & mask;
                let down = (current >> shift_2) & mask;
                current |= up | down;
            }
            let shift_4 = STRIDE_Y * 4;
            if shift_4 < 64 {
                let up = (current << shift_4) & mask;
                let down = (current >> shift_4) & mask;
                current |= up | down;
            }
            let shift_8 = STRIDE_Y * 8;
            if shift_8 < 64 {
                let up = (current << shift_8) & mask;
                let down = (current >> shift_8) & mask;
                current |= up | down;
            }
            let shift_16 = STRIDE_Y * 16;
            if shift_16 < 64 {
                let up = (current << shift_16) & mask;
                let down = (current >> shift_16) & mask;
                current |= up | down;
            }
            let shift_32 = STRIDE_Y * 32;
            if shift_32 < 64 {
                let up = (current << shift_32) & mask;
                let down = (current >> shift_32) & mask;
                current |= up | down;
            }

            spread_boundary = current;
        }

        // Intra-block vertical unions (polychromatic)
        if STRIDE_Y < 64 {
            let vertical_pairs = spread_boundary & (spread_boundary >> STRIDE_Y);
            if vertical_pairs != 0
                && self.merge_shifted(vertical_pairs, base_global, STRIDE_Y as isize, base_global)
            {
                expanded = true;
            }
        }

        // Horizontal unions
        let horizontal = (spread_boundary & (spread_boundary << 1)) & !self.row_start_mask;
        if horizontal != 0
            && self.merge_shifted(horizontal, base_global, -1, base_global)
        {
            expanded = true;
        }

        // Internal boundary connections (holes)
        if STRIDE_Y < 64 {
            let internal_bottom_edge = spread_boundary & (!valid_mask >> STRIDE_Y);
            let internal_top_edge = spread_boundary & (!valid_mask << STRIDE_Y);

            if self.connect_boundary_4way_ilp(
                internal_bottom_edge | internal_top_edge,
                base_global,
                0,
                boundary_node,
            ) {
                expanded = true;
            }
        }

        let new_occupied = occupied | spread_boundary;
        if new_occupied != occupied {
            occupied = new_occupied;
            expanded = true;
            if !SILENT {
                self.push_next(blk_idx);
            }
        }

        // ============== NEIGHBOR PROCESSING (POLYCHROMATIC) ==============
        // Up neighbor
        if blk_idx > 0 {
            let blk_up = blk_idx - 1;
            let (valid_up, occupied_up, effective_up) = {
                let b = self.blocks_state.get_unchecked(blk_up);
                (b.valid_mask, b.occupied, b.effective_mask)
            };
            let shift_amt = 64 - STRIDE_Y;
            let spread_to_up = spread_boundary << shift_amt;

            let boundary_connect_mask = spread_to_up & !valid_up;
            if boundary_connect_mask != 0
                && self.connect_boundary_4way_ilp(
                    boundary_connect_mask,
                    base_global,
                    -(shift_amt as isize),
                    boundary_node,
                )
            {
                expanded = true;
            }

            if valid_up != 0 {
                let grow_up = spread_to_up & !occupied_up & effective_up;
                if grow_up != 0 {
                    self.fast_grow_block::<SILENT>(blk_up, grow_up);
                    expanded = true;
                }

                let merge_up = spread_to_up & occupied_up;
                if self.merge_shifted(merge_up, base_global, -(shift_amt as isize), blk_up * 64) {
                    expanded = true;
                }
            }
        } else {
            // Top block - check boundary
            if STRIDE_Y < 64 {
                let mask_low = (1u64 << STRIDE_Y) - 1;
                let spread_boundary_masked = spread_boundary & mask_low;
                if self.connect_boundary_4way_ilp(
                    spread_boundary_masked,
                    base_global,
                    0,
                    boundary_node,
                ) {
                    expanded = true;
                }
            } else if STRIDE_Y == 64
                && self.connect_boundary_4way_ilp(spread_boundary, base_global, 0, boundary_node)
            {
                expanded = true;
            }
        }

        // Down neighbor
        if blk_idx + 1 < self.blocks_state.len() {
            let blk_down = blk_idx + 1;
            let (valid_down, occupied_down, effective_down) = {
                let b = self.blocks_state.get_unchecked(blk_down);
                (b.valid_mask, b.occupied, b.effective_mask)
            };
            let shift_amt = 64 - STRIDE_Y;
            let spread_to_down = spread_boundary >> shift_amt;

            let boundary_connect_mask = spread_to_down & !valid_down;
            if boundary_connect_mask != 0
                && self.connect_boundary_4way_ilp(
                    boundary_connect_mask,
                    base_global,
                    shift_amt as isize,
                    boundary_node,
                )
            {
                expanded = true;
            }

            if valid_down != 0 {
                let grow_down = spread_to_down & !occupied_down & effective_down;
                if grow_down != 0 {
                    self.fast_grow_block::<SILENT>(blk_down, grow_down);
                    expanded = true;
                }

                let merge_down = spread_to_down & occupied_down;
                if self.merge_shifted(merge_down, base_global, shift_amt as isize, blk_down * 64) {
                    expanded = true;
                }
            }
        } else {
            // Last block - bottom boundary
            if STRIDE_Y < 64 {
                let shift_amt = 64 - STRIDE_Y;
                let spread_boundary_masked = spread_boundary >> shift_amt;
                if self.connect_boundary_4way_ilp(
                    spread_boundary_masked,
                    base_global,
                    shift_amt as isize,
                    boundary_node,
                ) {
                    expanded = true;
                }
            } else if self.connect_boundary_4way_ilp(spread_boundary, base_global, 0, boundary_node)
            {
                expanded = true;
            }
        }

        // Horizontal edges
        let left_edge = spread_boundary & self.row_start_mask;
        if left_edge != 0
            && self.connect_boundary_4way_ilp(left_edge, base_global, 0, boundary_node)
        {
            expanded = true;
        }

        let right_edge = spread_boundary & self.row_end_mask;
        if right_edge != 0
            && self.connect_boundary_4way_ilp(right_edge, base_global, 0, boundary_node)
        {
            expanded = true;
        }

        // Clean up boundary
        boundary &= !spread_boundary;

        // Write back
        if occupied != initial_occupied || boundary != initial_boundary {
            let block = self.blocks_state.get_unchecked_mut(blk_idx);
            block.occupied = occupied;
            block.boundary = boundary;
            self.mark_block_dirty(blk_idx);
        }

        expanded
    }

    /// # Safety
    ///
    /// Caller must ensure:
    /// - `blk_idx` is within bounds of `self.blocks_state`
    /// - All internal state arrays are properly initialized
    #[inline(always)]
    pub unsafe fn process_block_small_stride<const SILENT: bool>(
        &mut self,
        blk_idx: usize,
    ) -> bool {
        // Assert supported stride (large stride module removed)
        debug_assert!(STRIDE_Y <= 64);

        // STRIDE_Y == 32 specialization (unchanged)
        if STRIDE_Y == 32 {
            let is_small = self.is_small_grid();
            let block_offset = self.parent_offset / 64;
            return Self::process_block_small_stride_32::<SILENT>(
                blk_idx,
                self.parents,
                self.blocks_state,
                self.defect_mask,
                self.block_dirty_mask,
                self.queued_mask,
                is_small,
                block_offset,
            );
        }

        let block = self.blocks_state.get_unchecked(blk_idx);
        if block.boundary == 0 {
            return false;
        }

        // Fast dispatch based on cached root
        let block_root = block.root;
        if block_root != u32::MAX {
            // Monochromatic fast path (>95% of blocks)
            return self.process_mono::<SILENT>(blk_idx, block_root);
        }

        // Check if we can determine monochromatic status
        let boundary = block.boundary;
        if boundary.count_ones() == 1 {
            // Single boundary bit - trivially monochromatic
            let base_global = self.parent_offset + blk_idx * 64;
            let first_bit = boundary.trailing_zeros() as usize;
            let root = self.find((base_global + first_bit) as u32);
            self.blocks_state.get_unchecked_mut(blk_idx).root = root;
            return self.process_mono::<SILENT>(blk_idx, root);
        }

        // Multiple boundary bits without cached root - check if monochromatic
        if boundary != 0 {
            let base_global = self.parent_offset + blk_idx * 64;
            let first_bit = boundary.trailing_zeros() as usize;
            let root_r = self.find((base_global + first_bit) as u32);
            let mut is_mono = true;

            let mut temp = boundary & !(1u64 << first_bit);
            while temp != 0 {
                let bit = temp.trailing_zeros() as usize;
                temp &= temp - 1;

                let node = (base_global + bit) as u32;
                // Fast path: direct parent check
                if *self.parents.get_unchecked(node as usize) == root_r {
                    continue;
                }
                // Slow path: full find
                if self.find(node) != root_r {
                    is_mono = false;
                    break;
                }
            }

            if is_mono {
                self.blocks_state.get_unchecked_mut(blk_idx).root = root_r;
                return self.process_mono::<SILENT>(blk_idx, root_r);
            }
        }

        // Polychromatic slow path
        self.process_poly::<SILENT>(blk_idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::decoder::state::{BoundaryConfig, DecodingState};
    use crate::topology::SquareGrid;

    #[test]
    fn test_small_stride_intra_block_parents() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Grid 32x32 (Stride 32). 1024 nodes. 16 blocks.
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 1 << 32;
            block.occupied = 1 << 32;
            block.valid_mask = !0;
            block.erasure_mask = 0;
        }

        unsafe {
            decoder.process_block_small_stride::<false>(0);
        }

        let root_32 = decoder.find(32);
        let root_0 = decoder.find(0);

        assert_eq!(root_32, root_0, "Node 32 should be connected to Node 0");

        let boundary = (decoder.parents.len() - 1) as u32;
        assert_eq!(root_0, boundary, "Node 0 should be connected to Boundary");
    }

    #[test]
    fn test_small_stride_horizontal_connectivity() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 32x32 grid. Stride 32.
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        // Activate bit 0. It should spread to bit 1 if erasure mask allows.
        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 1;
            block.occupied = 1;
            block.valid_mask = !0;
            block.erasure_mask = 0; // Allow spreading (0 = conductive)
        }

        unsafe {
            decoder.process_block_small_stride::<false>(0);
        }

        // Bit 0 should have spread to Bit 1, 2, ... 31
        let occupied = unsafe { decoder.blocks_state.get_unchecked(0).occupied };
        assert_ne!(occupied & 2, 0, "Bit 1 should be occupied");

        // Check connectivity
        let root0 = decoder.find(0);
        let root1 = decoder.find(1);

        assert_eq!(
            root0, root1,
            "Horizontal neighbors (0 and 1) should be connected"
        );
    }

    #[test]
    fn test_merge_mono_optimization_behavior() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        // Root source: Node 0
        let root_source = 0;

        unsafe {
            // Connect Node 2 to Root manually (Direct Parent)
            *decoder.parents.get_unchecked_mut(2) = root_source;

            // Verify initial state
            assert_ne!(decoder.find(1), decoder.find(root_source));
            assert_eq!(decoder.find(2), decoder.find(root_source));

            // merge_mono targeting Node 2 (should be no-op/not expanded for that bit)
            // Mask: bit 2
            let mask_only_connected = 1 << 2;
            let expanded = decoder.merge_mono(mask_only_connected, 0, root_source);
            assert!(!expanded, "Should not expand if already connected directly");

            // merge_mono targeting Node 1 (should expand)
            let mask_new = 1 << 1;
            let expanded_new = decoder.merge_mono(mask_new, 0, root_source);
            assert!(expanded_new, "Should expand for new node");
            assert_eq!(decoder.find(1), decoder.find(root_source));
        }
    }

    #[test]
    fn test_merge_mono_bulk_deduplication() {
        // Test that merge_mono deduplicates roots when multiple bits share the same root
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        let root_source = 100u32;

        unsafe {
            // Pre-connect nodes 0, 1, 2 to node 10 (so they share a root)
            *decoder.parents.get_unchecked_mut(0) = 10;
            *decoder.parents.get_unchecked_mut(1) = 10;
            *decoder.parents.get_unchecked_mut(2) = 10;
            // Node 10 is its own root
            *decoder.parents.get_unchecked_mut(10) = 10;

            // Ensure root_source is its own root
            *decoder.parents.get_unchecked_mut(root_source as usize) = root_source;

            // Merge nodes 0, 1, 2 (bits 0, 1, 2) with root_source
            // All three should deduplicate to a single union_roots(10, root_source)
            let mask = 0b111; // bits 0, 1, 2
            let expanded = decoder.merge_mono(mask, 0, root_source);

            assert!(expanded, "Should expand when merging");

            // All nodes should now be connected
            let final_root = decoder.find(0);
            assert_eq!(decoder.find(1), final_root);
            assert_eq!(decoder.find(2), final_root);
            assert_eq!(decoder.find(10), final_root);
            assert_eq!(decoder.find(root_source), final_root);
        }
    }

    #[test]
    fn test_merge_mono_skip_already_connected() {
        // Test that merge_mono skips nodes already connected to root_source
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        let root_source = 50u32;

        unsafe {
            // Make root_source its own root
            *decoder.parents.get_unchecked_mut(root_source as usize) = root_source;

            // Pre-connect nodes 0, 1 to root_source
            *decoder.parents.get_unchecked_mut(0) = root_source;
            *decoder.parents.get_unchecked_mut(1) = root_source;

            // merge_mono on bits 0, 1, 2 - only bit 2 should cause expansion
            let mask = 0b111;
            let expanded = decoder.merge_mono(mask, 0, root_source);

            assert!(expanded, "Should expand for node 2");
            assert_eq!(decoder.find(2), decoder.find(root_source));
        }
    }

    #[test]
    fn test_merge_mono_overflow_handling() {
        // Test that merge_mono handles more than 8 unique roots
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        let root_source = 200u32;

        unsafe {
            // Make 10 separate roots (nodes 0-9 are their own roots)
            for i in 0..10 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }
            *decoder.parents.get_unchecked_mut(root_source as usize) = root_source;

            // Merge all 10 nodes
            let mask = 0b1111111111; // bits 0-9
            let expanded = decoder.merge_mono(mask, 0, root_source);

            assert!(expanded);

            // All should be connected to root_source
            let final_root = decoder.find(root_source);
            for i in 0..10 {
                assert_eq!(
                    decoder.find(i as u32),
                    final_root,
                    "Node {} should be connected",
                    i
                );
            }
        }
    }

    #[test]
    fn test_vertical_neighbor_mono_to_mono_merge() {
        // Test the optimization: when both blocks are monochromatic, use union_roots
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Use stride 16 to test the generic small stride path
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // For stride 16, inter-block vertical: block N's top row spreads to block N-1's bottom row
            // Block layout: each block has 64 bits = 4 rows of 16 bits each
            // Row 0: bits 0-15, Row 1: bits 16-31, Row 2: bits 32-47, Row 3: bits 48-63
            // Spreading to Up block: boundary << (64-16) = boundary << 48
            // So for block 1 to spread to block 0, block 1 needs bits in its Row 0 (bits 0-15)

            // Block 1 setup - monochromatic with root 64, has boundary in Row 0
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0 set (bits 0-15 in block)
            block1.occupied = 0xFFFF;
            block1.root = 64; // Monochromatic
            block1.effective_mask = !0;
            block1.valid_mask = !0;
            block1.erasure_mask = 0;

            // Connect nodes 64-79 to root 64
            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            // Block 0 setup - monochromatic with root 0, has occupied in Row 3 (where Up spread lands)
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0;
            block0.occupied = 0xFFFF_0000_0000_0000u64; // Row 3 set (bits 48-63 in block)
            block0.root = 0; // Monochromatic
            block0.effective_mask = !0;
            block0.valid_mask = !0;
            block0.erasure_mask = 0;

            // Connect nodes 48-63 to root 0
            *decoder.parents.get_unchecked_mut(0) = 0;
            for i in 48..64 {
                *decoder.parents.get_unchecked_mut(i) = 0;
            }

            // Process block 1 - should spread up to block 0 and merge using mono-to-mono path
            decoder.process_block_small_stride::<false>(1);

            // Both blocks should be connected through their roots
            let root0 = decoder.find(0);
            let root64 = decoder.find(64);
            assert_eq!(
                root0, root64,
                "Mono-to-mono merge should connect the blocks"
            );
        }
    }

    #[test]
    fn test_merge_mono_empty_mask() {
        // Test merge_mono returns false when mask == 0 (line 14)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            let expanded = decoder.merge_mono(0, 0, 0);
            assert!(!expanded, "merge_mono with empty mask should return false");
        }
    }

    #[test]
    fn test_process_mono_zero_boundary() {
        // Test process_mono returns false when boundary == 0 (line 133)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 0; // Zero boundary
            block.occupied = 1;
            block.root = 0;
            block.valid_mask = !0;
            block.effective_mask = !0;

            let expanded = decoder.process_mono::<false>(0, 0);
            assert!(!expanded, "process_mono with zero boundary should return false");
        }
    }

    #[test]
    fn test_process_poly_zero_boundary() {
        // Test process_poly returns false when boundary == 0 (line 577)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 0; // Zero boundary
            block.occupied = 1;
            block.root = u32::MAX; // Polychromatic
            block.valid_mask = !0;
            block.effective_mask = !0;

            let expanded = decoder.process_poly::<false>(0);
            assert!(!expanded, "process_poly with zero boundary should return false");
        }
    }

    #[test]
    fn test_process_mono_internal_hole_connection() {
        // Test internal boundary/hole connection (line 244-251)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Create a block with a hole in the valid_mask
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 1 << 8; // Bit 8 (row 1)
            block.occupied = 1 << 8;
            block.root = 8;
            // Create a hole: valid_mask has bit 0 cleared (position 0 is invalid)
            block.valid_mask = !1u64; // All valid except bit 0
            block.effective_mask = !1u64;

            *decoder.parents.get_unchecked_mut(8) = 8;

            // Process - should hit internal hole connection
            let expanded = decoder.process_mono::<false>(0, 8);

            // Should have connected to boundary through the hole
            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root8 = decoder.find(8);
            assert_eq!(root8, boundary_node, "Should connect to boundary via hole");
            assert!(expanded);
        }
    }

    #[test]
    fn test_process_mono_up_neighbor_boundary_connection() {
        // Test up neighbor boundary connection when hitting invalid bits (lines 286-291)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Use larger grid with matching stride (16x16 -> STRIDE_Y=16, 4 rows/block -> 4 blocks)
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 1 spreads up to block 0
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0 (bits 0-15)
            block1.occupied = 0xFFFF;
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 64..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            // Block 0 has invalid bits where spread lands
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0;
            block0.occupied = 0;
            block0.root = u32::MAX;
            block0.valid_mask = 0x0000_FFFF_FFFF_FFFFu64; // Top row invalid
            block0.effective_mask = 0x0000_FFFF_FFFF_FFFFu64;

            decoder.process_mono::<false>(1, 64);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root64 = decoder.find(64);
            assert_eq!(root64, boundary_node, "Should connect to boundary via invalid up neighbor");
        }
    }

    #[test]
    fn test_process_mono_down_neighbor_different_clusters() {
        // Test down neighbor with different clusters (lines 438-453)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 0 spreads down to block 1
            // For stride 16: 4 rows per block, row 3 = bits 48-63
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.occupied = 0xFFFF_0000_0000_0000u64;
            block0.root = 48;
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(48) = 48;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(i) = 48;
            }

            // Block 1 has different root
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0;
            block1.occupied = 0xFFFF; // Row 0
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            decoder.process_mono::<false>(0, 48);

            // Both clusters should be merged
            let root48 = decoder.find(48);
            let root64 = decoder.find(64);
            assert_eq!(root48, root64, "Different clusters should merge");
        }
    }

    #[test]
    fn test_process_mono_top_block_boundary() {
        // Test top block boundary check (lines 356-377)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block 0 is the top block
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFF; // Row 0 at global top
            block0.occupied = 0xFF;
            block0.root = 0;
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(0) = 0;
            for i in 1..8 {
                *decoder.parents.get_unchecked_mut(i) = 0;
            }

            decoder.process_mono::<false>(0, 0);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root0 = decoder.find(0);
            assert_eq!(root0, boundary_node, "Top row should connect to boundary");
        }
    }

    #[test]
    fn test_process_mono_last_block_bottom_boundary() {
        // Test last block bottom boundary (lines 478-494)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Use 16x16 grid with STRIDE_Y=16, which gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        let num_blocks = decoder.blocks_state.len();
        assert!(num_blocks >= 3, "Need at least 3 blocks for this test");
        // Use second-to-last block (block 3 for 16x16) which has actual data nodes
        // Block 4 only contains the sentinel node
        let last_data_blk = num_blocks - 2;
        let base = last_data_blk * 64;

        // Verify indices are within bounds
        let parents_len = decoder.parents.len();
        assert!(base + 63 < parents_len, "Last node of last data block must fit in parents array");

        unsafe {
            // Last data block has boundary at bottom row
            // For stride 16: 4 rows per block, so row 3 = bits 48-63
            let block = decoder.blocks_state.get_unchecked_mut(last_data_blk);
            block.boundary = 0xFFFF_0000_0000_0000u64; // Row 3
            block.occupied = 0xFFFF_0000_0000_0000u64;
            block.root = (base + 48) as u32;
            block.valid_mask = !0;
            block.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(base + 48) = (base + 48) as u32;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(base + i) = (base + 48) as u32;
            }

            decoder.process_mono::<false>(last_data_blk, (base + 48) as u32);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root = decoder.find((base + 48) as u32);
            assert_eq!(root, boundary_node, "Bottom row of last data block should connect to boundary");
        }
    }

    #[test]
    fn test_process_mono_left_edge_boundary() {
        // Test left edge boundary check (lines 498-508)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block with left edge active
            let block = decoder.blocks_state.get_unchecked_mut(0);
            // row_start_mask for stride 8 includes bits 0, 8, 16, 24, 32, 40, 48, 56
            block.boundary = 0x0101_0101_0101_0101u64; // Left column
            block.occupied = 0x0101_0101_0101_0101u64;
            block.root = 0;
            block.valid_mask = !0;
            block.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(0) = 0;

            decoder.process_mono::<false>(0, 0);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root0 = decoder.find(0);
            assert_eq!(root0, boundary_node, "Left edge should connect to boundary");
        }
    }

    #[test]
    fn test_process_mono_right_edge_boundary() {
        // Test right edge boundary check (lines 510-520)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block with right edge active
            let block = decoder.blocks_state.get_unchecked_mut(0);
            // row_end_mask for stride 8 includes bits 7, 15, 23, 31, 39, 47, 55, 63
            block.boundary = 0x8080_8080_8080_8080u64; // Right column
            block.occupied = 0x8080_8080_8080_8080u64;
            block.root = 7;
            block.valid_mask = !0;
            block.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(7) = 7;

            decoder.process_mono::<false>(0, 7);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root7 = decoder.find(7);
            assert_eq!(root7, boundary_node, "Right edge should connect to boundary");
        }
    }

    #[test]
    fn test_process_mono_up_neighbor_poly() {
        // Test up neighbor polychromatic path (line 341-352)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 1 is mono, spreads up to poly block 0
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0
            block1.occupied = 0xFFFF;
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            // Block 0 is polychromatic (root = MAX)
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0;
            block0.occupied = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.root = u32::MAX; // Polychromatic
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            // Make nodes in row 3 have different roots
            for i in 48..64 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }

            decoder.process_mono::<false>(1, 64);

            // Block 1's cluster should merge with block 0's nodes
            let root64 = decoder.find(64);
            let root48 = decoder.find(48);
            assert_eq!(root64, root48, "Mono should merge with poly neighbor");
        }
    }

    #[test]
    fn test_process_mono_down_neighbor_poly() {
        // Test down neighbor polychromatic path (lines 457-469)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 0 is mono, spreads down to poly block 1
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.occupied = 0xFFFF_0000_0000_0000u64;
            block0.root = 48;
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(48) = 48;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(i) = 48;
            }

            // Block 1 is polychromatic
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0;
            block1.occupied = 0xFFFF; // Row 0
            block1.root = u32::MAX; // Polychromatic
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            // Make nodes in row 0 of block 1 have different roots
            for i in 64..80 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }

            decoder.process_mono::<false>(0, 48);

            // Block 0's cluster should merge with block 1's nodes
            let root48 = decoder.find(48);
            let root64 = decoder.find(64);
            assert_eq!(root48, root64, "Mono should merge with poly down neighbor");
        }
    }

    #[test]
    fn test_process_poly_horizontal_unions() {
        // Test horizontal unions in process_poly (line 658-663)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block 0 is polychromatic with horizontal neighbors
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0b11; // Bits 0 and 1 adjacent horizontally
            block0.occupied = 0b11;
            block0.root = u32::MAX; // Polychromatic
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            // Different roots for adjacent nodes
            *decoder.parents.get_unchecked_mut(0) = 0;
            *decoder.parents.get_unchecked_mut(1) = 1;

            decoder.process_poly::<false>(0);

            // Should have merged horizontally
            let root0 = decoder.find(0);
            let root1 = decoder.find(1);
            assert_eq!(root0, root1, "Horizontal neighbors should merge in poly path");
        }
    }

    #[test]
    fn test_process_poly_up_neighbor_boundary() {
        // Test poly up neighbor boundary connection (lines 700-710)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 1 is poly, spreads up to invalid region
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0
            block1.occupied = 0xFFFF;
            block1.root = u32::MAX; // Polychromatic
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            for i in 64..80 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }

            // Block 0 has invalid bits where spread lands
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.valid_mask = 0x0000_FFFF_FFFF_FFFFu64; // Top row invalid
            block0.effective_mask = 0x0000_FFFF_FFFF_FFFFu64;
            block0.occupied = 0;
            block0.boundary = 0;
            block0.root = u32::MAX;

            decoder.process_poly::<false>(1);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root64 = decoder.find(64);
            assert_eq!(root64, boundary_node, "Poly should connect to boundary via invalid up");
        }
    }

    #[test]
    fn test_process_poly_down_neighbor_boundary() {
        // Test poly down neighbor boundary connection (lines 754-764)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 0 is poly, spreads down to invalid region
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.occupied = 0xFFFF_0000_0000_0000u64;
            block0.root = u32::MAX; // Polychromatic
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            for i in 48..64 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }

            // Block 1 has invalid bits where spread lands (row 0 = bits 0-15)
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.valid_mask = 0xFFFF_FFFF_FFFF_0000u64; // Bottom row invalid
            block1.effective_mask = 0xFFFF_FFFF_FFFF_0000u64;
            block1.occupied = 0;
            block1.boundary = 0;
            block1.root = u32::MAX;

            decoder.process_poly::<false>(0);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root48 = decoder.find(48);
            assert_eq!(root48, boundary_node, "Poly should connect to boundary via invalid down");
        }
    }

    #[test]
    fn test_process_poly_left_right_edges() {
        // Test poly left/right edge boundary (lines 798-811)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block with left and right edges active
            let block = decoder.blocks_state.get_unchecked_mut(0);
            // Left column: bits 0, 8, 16, 24, 32, 40, 48, 56
            // Right column: bits 7, 15, 23, 31, 39, 47, 55, 63
            block.boundary = 0x8181_8181_8181_8181u64; // Left and right columns
            block.occupied = 0x8181_8181_8181_8181u64;
            block.root = u32::MAX; // Polychromatic
            block.valid_mask = !0;
            block.effective_mask = !0;

            // Make each edge node its own root
            for bit in [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63] {
                *decoder.parents.get_unchecked_mut(bit) = bit as u32;
            }

            decoder.process_poly::<false>(0);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root0 = decoder.find(0); // Left edge
            let root7 = decoder.find(7); // Right edge
            assert_eq!(root0, boundary_node, "Left edge should connect to boundary in poly");
            assert_eq!(root7, boundary_node, "Right edge should connect to boundary in poly");
        }
    }

    #[test]
    fn test_process_poly_top_block_boundary() {
        // Test poly top block boundary (lines 724-742)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Block 0 is the top block, polychromatic
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFF; // Row 0 at global top
            block0.occupied = 0xFF;
            block0.root = u32::MAX; // Polychromatic
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            for i in 0..8 {
                *decoder.parents.get_unchecked_mut(i) = i as u32;
            }

            decoder.process_poly::<false>(0);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root0 = decoder.find(0);
            assert_eq!(root0, boundary_node, "Top row should connect to boundary in poly");
        }
    }

    #[test]
    fn test_process_poly_bottom_block_boundary() {
        // Test poly bottom block boundary (lines 778-796)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        let num_blocks = decoder.blocks_state.len();
        assert!(num_blocks >= 3, "Need at least 3 blocks");
        // Use second-to-last block (block 3 for 16x16) which has actual data nodes
        // Block 4 only contains the sentinel node
        let last_data_blk = num_blocks - 2;
        let base = last_data_blk * 64;

        // Verify indices are within bounds
        let parents_len = decoder.parents.len();
        assert!(base + 63 < parents_len, "Last node of last data block must fit in parents array");

        unsafe {
            // Last data block is polychromatic with bottom boundary (row 3 = bits 48-63)
            let block = decoder.blocks_state.get_unchecked_mut(last_data_blk);
            block.boundary = 0xFFFF_0000_0000_0000u64;
            block.occupied = 0xFFFF_0000_0000_0000u64;
            block.root = u32::MAX; // Polychromatic
            block.valid_mask = !0;
            block.effective_mask = !0;

            for i in 48..64 {
                *decoder.parents.get_unchecked_mut(base + i) = (base + i) as u32;
            }

            decoder.process_poly::<false>(last_data_blk);

            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root = decoder.find((base + 48) as u32);
            assert_eq!(root, boundary_node, "Bottom row of last data block should connect to boundary in poly");
        }
    }

    #[test]
    fn test_process_mono_cached_root_invalidation() {
        // Test cached root invalidation path (lines 140-145)
        // Use a 16x16 grid with STRIDE_Y=16 (not 32) to test the generic path
        // Block 1 is interior (not at top/bottom), neighbors are valid
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        // Disable horizontal boundary checks
        decoder.boundary_config = BoundaryConfig {
            check_top: true,    // Won't trigger because block 1 is not at top
            check_bottom: true, // Won't trigger because block 1 is not at bottom
            check_left: false,
            check_right: false,
        };

        unsafe {
            // Use block 1 (nodes 64-127, rows 4-7)
            // Interior position: row 4, col 1 = node 4*16+1 = 65, which is bit 1 in block 1
            // Stale root points to node 65, actual root at node 66
            let block = decoder.blocks_state.get_unchecked_mut(1);
            block.boundary = 1 << 1; // bit 1 = node 65
            block.occupied = 1 << 1;
            block.root = 65; // Cached root points to node 65
            block.valid_mask = !0;
            block.effective_mask = !0;

            // Node 65's parent is actually node 66 (cache is stale)
            *decoder.parents.get_unchecked_mut(65) = 66;
            *decoder.parents.get_unchecked_mut(66) = 66;

            // Process should detect stale cache and update
            decoder.process_mono::<false>(1, 65);

            // The find operation during process_mono should have resolved this
            let root65 = decoder.find(65);
            assert_eq!(root65, 66, "Should use actual root after cache invalidation");
        }
    }

    #[test]
    fn test_process_mono_up_neighbor_same_cluster_after_resolution() {
        // Test up neighbor same cluster after root resolution (lines 318-323)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 1 mono spreads up to block 0 mono
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0
            block1.occupied = 0xFFFF;
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            // Block 0 has a cached root that needs resolution
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0;
            block0.occupied = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.root = 48; // Cached root
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            // But 48's actual root is 64 (same cluster)
            *decoder.parents.get_unchecked_mut(48) = 64;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            decoder.process_mono::<false>(1, 64);

            // Should have recognized same cluster and just grown
            let root48 = decoder.find(48);
            let root64 = decoder.find(64);
            assert_eq!(root48, root64);
        }
    }

    #[test]
    fn test_process_mono_down_neighbor_same_cluster_after_resolution() {
        // Test down neighbor same cluster after root resolution (lines 435-440)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16 gives 4 blocks
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Block 0 mono spreads down to block 1 mono
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bits 48-63)
            block0.occupied = 0xFFFF_0000_0000_0000u64;
            block0.root = 48;
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(48) = 48;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(i) = 48;
            }

            // Block 1 has cached root that resolves to same cluster
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0;
            block1.occupied = 0xFFFF; // Row 0
            block1.root = 64; // Cached root
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            // But 64's actual root is 48 (same cluster)
            *decoder.parents.get_unchecked_mut(64) = 48;
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 48;
            }

            decoder.process_mono::<false>(0, 48);

            // Should have recognized same cluster
            let root48 = decoder.find(48);
            let root64 = decoder.find(64);
            assert_eq!(root48, root64);
        }
    }

    #[test]
    fn test_very_small_stride_deep_shifts() {
        // Test shift_16 and shift_32 paths with STRIDE_Y=2 (lines 192-203)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // For STRIDE_Y=2, max(width, height) must be <= 2
        // 2x2 grid: 4 nodes = fits in 1 block
        let mut decoder = DecodingState::<SquareGrid, 2>::new(&mut arena, 2, 2, 1);

        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 1; // Single bit at position 0
            block.occupied = 1;
            block.root = 0;
            block.valid_mask = 0b1111; // Only 4 bits valid for 2x2 grid
            block.effective_mask = 0b1111;

            *decoder.parents.get_unchecked_mut(0) = 0;

            decoder.process_mono::<false>(0, 0);

            // Should have spread vertically through shift_2
            // With stride 2, vertical spread to bit 2 (0 + stride)
            let occupied = decoder.blocks_state.get_unchecked(0).occupied;
            assert!(occupied & (1 << 2) != 0 || occupied & (1 << 1) != 0,
                "Should spread with small stride");
        }
    }

    #[test]
    fn test_process_block_dispatch_to_poly() {
        // Test process_block_small_stride dispatching to process_poly (line 905)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // Setup block as definitely polychromatic (multiple boundary bits with different roots)
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 0b11; // Two adjacent bits
            block.occupied = 0b11;
            block.root = u32::MAX; // Polychromatic marker
            block.valid_mask = !0;
            block.effective_mask = !0;

            // Make them have different roots
            *decoder.parents.get_unchecked_mut(0) = 0;
            *decoder.parents.get_unchecked_mut(1) = 1;

            // Call the dispatch function
            let expanded = decoder.process_block_small_stride::<false>(0);

            // Should have processed and merged
            assert!(expanded);
            let root0 = decoder.find(0);
            let root1 = decoder.find(1);
            assert_eq!(root0, root1, "Poly path should merge adjacent nodes");
        }
    }

    #[test]
    fn test_process_block_single_boundary_bit_mono_conversion() {
        // Test single boundary bit becomes mono (lines 864-871)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 1; // Single bit
            block.occupied = 1;
            block.root = u32::MAX; // Unknown (will be determined)
            block.valid_mask = !0;
            block.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(0) = 0;

            decoder.process_block_small_stride::<false>(0);

            // Block should now be marked as monochromatic
            let new_root = decoder.blocks_state.get_unchecked(0).root;
            assert_ne!(new_root, u32::MAX, "Single boundary bit should convert to mono");
        }
    }

    #[test]
    fn test_process_block_multi_boundary_mono_check() {
        // Test multiple boundary bits with same root becomes mono (lines 875-901)
        // Use a 16x16 grid with STRIDE_Y=16 (not 32) to test the generic path
        // Block 1 is interior (not at top/bottom)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        // Disable horizontal boundary checks
        decoder.boundary_config = BoundaryConfig {
            check_top: true,    // Won't trigger because block 1 is not at top
            check_bottom: true, // Won't trigger because block 1 is not at bottom
            check_left: false,
            check_right: false,
        };

        unsafe {
            // Use block 1 (nodes 64-127, rows 4-7)
            // Use three adjacent nodes at row 4, cols 1-3: nodes 65, 66, 67 = bits 1, 2, 3
            let block = decoder.blocks_state.get_unchecked_mut(1);
            block.boundary = 0b1110; // Three bits at positions 1, 2, 3 (nodes 65-67)
            block.occupied = 0b1110;
            block.root = u32::MAX; // Unknown (poly initially)
            block.valid_mask = !0;
            block.effective_mask = !0;

            // All three share the same root (node 65)
            *decoder.parents.get_unchecked_mut(65) = 65;
            *decoder.parents.get_unchecked_mut(66) = 65;
            *decoder.parents.get_unchecked_mut(67) = 65;

            decoder.process_block_small_stride::<false>(1);

            // Block should be recognized as monochromatic
            let new_root = decoder.blocks_state.get_unchecked(1).root;
            assert_eq!(new_root, 65, "All same root should be recognized as mono");
        }
    }

    #[test]
    fn test_process_mono_up_neighbor_same_cluster_grow() {
        // Test up neighbor same cluster fast path with grow (lines 303-306)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        // Disable boundary checks
        decoder.boundary_config = BoundaryConfig {
            check_top: false,
            check_bottom: false,
            check_left: false,
            check_right: false,
        };

        unsafe {
            // Block 1 has boundary at row 0 (spreading up to block 0)
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0 (bits 0-15)
            block1.occupied = 0xFFFF;
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            // Block 0 (up neighbor) has same root but some unoccupied space
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bottom of block 0)
            block0.occupied = 0xFFFF_0000_0000_0000u64; // Only row 3 occupied
            block0.root = 64; // Same root!
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            // Set up parents - both share root 64
            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 48..64 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            let expanded = decoder.process_mono::<false>(1, 64);

            // Processing should expand (grow into block 0's unoccupied space)
            assert!(expanded, "Should expand when growing into same-cluster neighbor");
        }
    }

    #[test]
    fn test_process_mono_last_block_bottom_boundary_check() {
        // Test bottom boundary check for last block (lines 473-484)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        // Enable bottom boundary check
        decoder.boundary_config = BoundaryConfig {
            check_top: false,
            check_bottom: true, // Enable bottom boundary
            check_left: false,
            check_right: false,
        };

        unsafe {
            let num_blocks = decoder.blocks_state.len();
            // Use the last data block (block 3 for 16x16 grid)
            let last_data_blk = num_blocks - 2;
            let base = last_data_blk * 64;

            // Last block has boundary at bottom row
            let block = decoder.blocks_state.get_unchecked_mut(last_data_blk);
            block.boundary = 0xFFFF_0000_0000_0000u64; // Row 3 (bottom of block)
            block.occupied = 0xFFFF_0000_0000_0000u64;
            block.root = (base + 48) as u32;
            block.valid_mask = !0;
            block.effective_mask = !0;

            *decoder.parents.get_unchecked_mut(base + 48) = (base + 48) as u32;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(base + i) = (base + 48) as u32;
            }

            decoder.process_mono::<false>(last_data_blk, (base + 48) as u32);

            // Should connect to boundary node
            let boundary_node = (decoder.parents.len() - 1) as u32;
            let root = decoder.find((base + 48) as u32);
            assert_eq!(root, boundary_node, "Last block bottom row should connect to boundary");
        }
    }

    #[test]
    fn test_process_mono_different_clusters_grow() {
        // Test different clusters grow path (lines 334-337)
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // 16x16 grid with STRIDE_Y=16
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        // Disable boundary checks
        decoder.boundary_config = BoundaryConfig {
            check_top: false,
            check_bottom: false,
            check_left: false,
            check_right: false,
        };

        unsafe {
            // Block 1 spreading up to block 0 with different root
            let block1 = decoder.blocks_state.get_unchecked_mut(1);
            block1.boundary = 0xFFFF; // Row 0
            block1.occupied = 0xFFFF;
            block1.root = 64;
            block1.valid_mask = !0;
            block1.effective_mask = !0;

            // Block 0 has different root and unoccupied space for grow
            let block0 = decoder.blocks_state.get_unchecked_mut(0);
            block0.boundary = 0xFFFF_0000_0000_0000u64; // Row 3
            block0.occupied = 0xFFFF_0000_0000_0000u64; // Only row 3 occupied
            block0.root = 48; // Different root!
            block0.valid_mask = !0;
            block0.effective_mask = !0;

            // Set up different roots
            *decoder.parents.get_unchecked_mut(48) = 48;
            *decoder.parents.get_unchecked_mut(64) = 64;
            for i in 49..64 {
                *decoder.parents.get_unchecked_mut(i) = 48;
            }
            for i in 65..80 {
                *decoder.parents.get_unchecked_mut(i) = 64;
            }

            let expanded = decoder.process_mono::<false>(1, 64);

            // Should expand (union different clusters and grow)
            assert!(expanded, "Should expand when merging different clusters");
            // Clusters should be merged
            let root48 = decoder.find(48);
            let root64 = decoder.find(64);
            assert_eq!(root48, root64, "Different clusters should be merged");
        }
    }
}
