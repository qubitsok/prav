#![allow(unsafe_op_in_unsafe_fn)]
use crate::decoder::state::DecodingState;
use crate::intrinsics::{blsr, spread_syndrome_masked, tzcnt};
use crate::topology::Topology;

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    /// Unrolled processing for exactly 16 blocks (1024 nodes) with STRIDE_Y=32.
    /// Returns (any_expanded, next_active_mask).
    ///
    /// # Safety
    ///
    /// Caller must ensure the decoder state is properly initialized with exactly 16 blocks.
    pub unsafe fn process_all_blocks_stride_32_unrolled_16(&mut self) -> (bool, u64) {
        const ROW_START_MASK: u64 = 0x0000000100000001;
        const ROW_END_MASK: u64 = 0x8000000080000000;

        let parents_ptr = self.parents.as_mut_ptr();
        let parents_len = self.parents.len();
        let boundary_node = (parents_len - 1) as u32;

        let blocks_state_ptr = self.blocks_state.as_mut_ptr();
        let block_dirty_mask_ptr = self.block_dirty_mask.as_mut_ptr();

        let active_mask = self.active_block_mask;
        let mut next_mask = 0u64;
        let mut any_expanded = false;

        macro_rules! mark_dirty {
            ($blk:expr) => {
                let mask_idx = $blk >> 6;
                let mask_bit = $blk & 63;
                let m_ptr = block_dirty_mask_ptr.add(mask_idx);
                *m_ptr |= 1 << mask_bit;
            };
        }

        macro_rules! union_boundary {
            ($u:expr) => {
                let mut root_u = *parents_ptr.add($u as usize);
                while root_u != *parents_ptr.add(root_u as usize) {
                    let gp = *parents_ptr.add(root_u as usize);
                    *parents_ptr.add(root_u as usize) = gp;
                    mark_dirty!(root_u as usize >> 6);
                    root_u = gp;
                }

                if root_u != boundary_node {
                    if root_u < boundary_node {
                        *parents_ptr.add(root_u as usize) = boundary_node;
                        mark_dirty!(root_u as usize >> 6);
                    } else {
                        *parents_ptr.add(boundary_node as usize) = root_u;
                        mark_dirty!(boundary_node as usize >> 6);
                    }
                    any_expanded = true;
                }
            };
        }

        macro_rules! connect_bits {
            ($mask:expr, $base_global:expr, $offset:expr) => {
                let mut m = $mask;
                while m != 0 {
                    let bit = tzcnt(m);
                    m = blsr(m);
                    let u = (($base_global as isize) + ($offset as isize) + (bit as isize)) as u32;
                    union_boundary!(u);
                }
            };
        }

        macro_rules! fast_grow {
            ($blk:expr, $grow_mask:expr) => {
                let g = $grow_mask;
                if g != 0 {
                    mark_dirty!($blk);
                    (*blocks_state_ptr.add($blk)).occupied |= g;
                    (*blocks_state_ptr.add($blk)).boundary |= g;
                    next_mask |= 1 << $blk;
                }
            };
        }

        macro_rules! merge_shifted_macro {
            ($mask:expr, $base_src:expr, $shift:expr, $base_target:expr) => {
                let mut m = $mask;
                while m != 0 {
                    let start_bit = m.trailing_zeros();
                    let shifted = m >> start_bit;
                    let run_len = (!shifted).trailing_zeros();

                    if run_len == 64 {
                        m = 0;
                    } else {
                        m ^= ((1u64 << run_len) - 1) << start_bit;
                    }

                    let base_u = ($base_src as isize + start_bit as isize + $shift) as u32;
                    let base_v = ($base_target + start_bit as usize) as u32;

                    for k in 0..run_len {
                        let u = base_u + k;
                        let v = base_v + k;
                        let pu = *parents_ptr.add(u as usize);
                        let pv = *parents_ptr.add(v as usize);
                        if pu == u && pv == v {
                            if u != v {
                                if u < v {
                                    *parents_ptr.add(u as usize) = v;
                                    mark_dirty!(u as usize >> 6);
                                } else {
                                    *parents_ptr.add(v as usize) = u;
                                    mark_dirty!(v as usize >> 6);
                                }
                                any_expanded = true;
                            }
                        } else {
                            let mut root_u = pu;
                            while root_u != *parents_ptr.add(root_u as usize) {
                                let gp = *parents_ptr.add(root_u as usize);
                                *parents_ptr.add(root_u as usize) = gp;
                                mark_dirty!(root_u as usize >> 6);
                                root_u = gp;
                            }
                            let mut root_v = pv;
                            while root_v != *parents_ptr.add(root_v as usize) {
                                let gp = *parents_ptr.add(root_v as usize);
                                *parents_ptr.add(root_v as usize) = gp;
                                mark_dirty!(root_v as usize >> 6);
                                root_v = gp;
                            }
                            if root_u != root_v {
                                if root_u < root_v {
                                    *parents_ptr.add(root_u as usize) = root_v;
                                    mark_dirty!(root_u as usize >> 6);
                                } else {
                                    *parents_ptr.add(root_v as usize) = root_u;
                                    mark_dirty!(root_v as usize >> 6);
                                }
                                any_expanded = true;
                            }
                        }
                    }
                }
            };
        }

        macro_rules! process_block_unrolled {
            ($blk:expr, $up_type:ident, $down_type:ident) => {
                if (active_mask & (1 << $blk)) != 0 {
                    const BLK: usize = $blk;
                    const BASE_GLOBAL: usize = BLK * 64;

                    let mut boundary = (*blocks_state_ptr.add(BLK)).boundary;

                    if boundary != 0 {
                        let mut occupied = (*blocks_state_ptr.add(BLK)).occupied;
                        let initial_occupied = occupied;
                        let initial_boundary = boundary;

                        let valid_mask = (*blocks_state_ptr.add(BLK)).valid_mask;
                        // Use effective_mask from Hot state
                        let effective_mask = (*blocks_state_ptr.add(BLK)).effective_mask;
                        let mask = effective_mask;

                        let mut spread_boundary = spread_syndrome_masked(boundary, mask, ROW_END_MASK, ROW_START_MASK);
                        {
                            let up = (spread_boundary << 32) & mask;
                            let down = (spread_boundary >> 32) & mask;
                            spread_boundary |= up | down;
                        }

                        let internal_bottom_edge = spread_boundary & (!valid_mask >> 32);
                        let internal_top_edge = spread_boundary & (!valid_mask << 32);
                        let internal_boundary_mask = internal_bottom_edge | internal_top_edge;
                        if internal_boundary_mask != 0 {
                            connect_bits!(internal_boundary_mask, BASE_GLOBAL, 0);
                        }

                        // Intra-block Vertical Union
                        let vertical_pairs = spread_boundary & (spread_boundary >> 32) & 0xFFFFFFFF;
                        if vertical_pairs != 0 {
                            merge_shifted_macro!(vertical_pairs, BASE_GLOBAL, 32, BASE_GLOBAL);
                        }

                        // Intra-block Horizontal Union
                        let horizontal_pairs = spread_boundary & (spread_boundary >> 1) & !ROW_END_MASK;
                        if horizontal_pairs != 0 {
                            merge_shifted_macro!(horizontal_pairs, BASE_GLOBAL, 1, BASE_GLOBAL);
                        }

                        let new_occupied = occupied | spread_boundary;
                        if new_occupied != occupied {
                            occupied = new_occupied;
                            any_expanded = true;
                            next_mask |= 1 << BLK;
                        }

                        process_block_unrolled!(@UP $up_type, spread_boundary, BASE_GLOBAL);
                        process_block_unrolled!(@DOWN $down_type, spread_boundary, BASE_GLOBAL);

                        let row_start_hits = spread_boundary & ROW_START_MASK;
                        if row_start_hits != 0 { connect_bits!(row_start_hits, BASE_GLOBAL, 0); }
                        let row_end_hits = spread_boundary & ROW_END_MASK;
                        if row_end_hits != 0 { connect_bits!(row_end_hits, BASE_GLOBAL, 0); }

                        boundary &= !spread_boundary;
                        if occupied != initial_occupied || boundary != initial_boundary {
                             (*blocks_state_ptr.add(BLK)).occupied = occupied;
                             (*blocks_state_ptr.add(BLK)).boundary = boundary;
                        }
                    }
                }
            };

            (@UP BOUNDARY, $spread_boundary:expr, $base_global:expr) => {
                let hits = $spread_boundary & 0xFFFFFFFF;
                if hits != 0 { connect_bits!(hits, $base_global, 0); }
            };
            (@UP NEIGHBOR, $spread_boundary:expr, $base_global:expr) => {
                const BLK_UP: usize = BLK - 1;
                let valid_up = (*blocks_state_ptr.add(BLK_UP)).valid_mask;
                let spread_to_up = $spread_boundary << 32;
                let boundary_hits = spread_to_up & !valid_up;
                if boundary_hits != 0 { connect_bits!(boundary_hits, $base_global, -32); }
                if valid_up != 0 {
                    let occupied_up = (*blocks_state_ptr.add(BLK_UP)).occupied;
                    let effective_up = (*blocks_state_ptr.add(BLK_UP)).effective_mask;
                    let grow_up = spread_to_up & !occupied_up & effective_up;
                    fast_grow!(BLK_UP, grow_up);
                    if grow_up != 0 { any_expanded = true; }
                    let merge_up = spread_to_up & occupied_up;
                    if merge_up != 0 { merge_shifted_macro!(merge_up, $base_global, -32, BLK_UP * 64); }
                }
            };

            (@DOWN BOUNDARY, $spread_boundary:expr, $base_global:expr) => {
                let hits = $spread_boundary >> 32;
                if hits != 0 { connect_bits!(hits, $base_global, 32); }
            };
            (@DOWN NEIGHBOR, $spread_boundary:expr, $base_global:expr) => {
                const BLK_DOWN: usize = BLK + 1;
                let valid_down = (*blocks_state_ptr.add(BLK_DOWN)).valid_mask;
                let spread_to_down = $spread_boundary >> 32;
                let boundary_hits = spread_to_down & !valid_down;
                if boundary_hits != 0 { connect_bits!(boundary_hits, $base_global, 32); }
                if valid_down != 0 {
                    let occupied_down = (*blocks_state_ptr.add(BLK_DOWN)).occupied;
                    let effective_down = (*blocks_state_ptr.add(BLK_DOWN)).effective_mask;
                    let grow_down = spread_to_down & !occupied_down & effective_down;
                    fast_grow!(BLK_DOWN, grow_down);
                    if grow_down != 0 { any_expanded = true; }
                    let merge_down = spread_to_down & occupied_down;
                    if merge_down != 0 { merge_shifted_macro!(merge_down, $base_global, 32, BLK_DOWN * 64); }
                }
            };
        }

        process_block_unrolled!(0, BOUNDARY, NEIGHBOR);
        process_block_unrolled!(1, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(2, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(3, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(4, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(5, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(6, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(7, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(8, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(9, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(10, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(11, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(12, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(13, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(14, NEIGHBOR, NEIGHBOR);
        process_block_unrolled!(15, NEIGHBOR, BOUNDARY);

        (any_expanded, next_mask)
    }

    /// Optimized processing for stride-64 grids (64x64).
    /// Returns (any_expanded, next_active_mask).
    ///
    /// # Safety
    ///
    /// Caller must ensure the decoder state is properly initialized for stride-64 processing.
    pub unsafe fn process_all_blocks_stride_64(&mut self) -> (bool, u64) {
        const ROW_START_MASK: u64 = 0x0000000000000001; // Bit 0 only
        const ROW_END_MASK: u64 = 0x8000000000000000; // Bit 63 only

        let parents_ptr = self.parents.as_mut_ptr();
        let parents_len = self.parents.len();
        let boundary_node = (parents_len - 1) as u32;

        let blocks_state_ptr = self.blocks_state.as_mut_ptr();
        let block_dirty_mask_ptr = self.block_dirty_mask.as_mut_ptr();

        let num_blocks = self.blocks_state.len().min(65); // Max 64 data blocks + boundary
        let active_mask = self.active_block_mask;
        let mut next_mask = 0u64;
        let mut any_expanded = false;

        macro_rules! mark_dirty {
            ($blk:expr) => {
                let mask_idx = $blk >> 6;
                let mask_bit = $blk & 63;
                let m_ptr = block_dirty_mask_ptr.add(mask_idx);
                *m_ptr |= 1 << mask_bit;
            };
        }

        macro_rules! union_boundary {
            ($u:expr) => {
                let mut root_u = *parents_ptr.add($u as usize);
                while root_u != *parents_ptr.add(root_u as usize) {
                    let gp = *parents_ptr.add(root_u as usize);
                    *parents_ptr.add(root_u as usize) = gp;
                    mark_dirty!(root_u as usize >> 6);
                    root_u = gp;
                }
                if root_u != boundary_node {
                    if root_u < boundary_node {
                        *parents_ptr.add(root_u as usize) = boundary_node;
                        mark_dirty!(root_u as usize >> 6);
                    } else {
                        *parents_ptr.add(boundary_node as usize) = root_u;
                        mark_dirty!(boundary_node as usize >> 6);
                    }
                    any_expanded = true;
                }
            };
        }

        macro_rules! connect_bits {
            ($mask:expr, $base_global:expr, $offset:expr) => {
                let mut m = $mask;
                while m != 0 {
                    let bit = tzcnt(m);
                    m = blsr(m);
                    let u = (($base_global as isize) + ($offset as isize) + (bit as isize)) as u32;
                    union_boundary!(u);
                }
            };
        }

        macro_rules! fast_grow {
            ($blk:expr, $grow_mask:expr) => {
                let g = $grow_mask;
                if g != 0 {
                    mark_dirty!($blk);
                    (*blocks_state_ptr.add($blk)).occupied |= g;
                    (*blocks_state_ptr.add($blk)).boundary |= g;
                    next_mask |= 1 << $blk;
                }
            };
        }

        macro_rules! merge_horizontal {
            ($mask:expr, $base:expr) => {
                let mut m = $mask;
                while m != 0 {
                    let start_bit = m.trailing_zeros();
                    let shifted = m >> start_bit;
                    let run_len = (!shifted).trailing_zeros();
                    if run_len == 64 {
                        m = 0;
                    } else {
                        m ^= ((1u64 << run_len) - 1) << start_bit;
                    }

                    let base_u = ($base + start_bit as usize + 1) as u32;
                    let base_v = ($base + start_bit as usize) as u32;

                    for k in 0..run_len {
                        let u = base_u + k;
                        let v = base_v + k;
                        let pu = *parents_ptr.add(u as usize);
                        let pv = *parents_ptr.add(v as usize);
                        if pu == u && pv == v {
                            if u != v {
                                if u < v {
                                    *parents_ptr.add(u as usize) = v;
                                    mark_dirty!(u as usize >> 6);
                                } else {
                                    *parents_ptr.add(v as usize) = u;
                                    mark_dirty!(v as usize >> 6);
                                }
                                any_expanded = true;
                            }
                        } else {
                            let mut root_u = pu;
                            while root_u != *parents_ptr.add(root_u as usize) {
                                let gp = *parents_ptr.add(root_u as usize);
                                *parents_ptr.add(root_u as usize) = gp;
                                mark_dirty!(root_u as usize >> 6);
                                root_u = gp;
                            }
                            let mut root_v = pv;
                            while root_v != *parents_ptr.add(root_v as usize) {
                                let gp = *parents_ptr.add(root_v as usize);
                                *parents_ptr.add(root_v as usize) = gp;
                                mark_dirty!(root_v as usize >> 6);
                                root_v = gp;
                            }
                            if root_u != root_v {
                                if root_u < root_v {
                                    *parents_ptr.add(root_u as usize) = root_v;
                                    mark_dirty!(root_u as usize >> 6);
                                } else {
                                    *parents_ptr.add(root_v as usize) = root_u;
                                    mark_dirty!(root_v as usize >> 6);
                                }
                                any_expanded = true;
                            }
                        }
                    }
                }
            };
        }

        // Process all active blocks
        let mut blk_mask = active_mask;
        while blk_mask != 0 {
            let blk = tzcnt(blk_mask) as usize;
            blk_mask = blsr(blk_mask);

            if blk >= num_blocks {
                continue;
            }

            let base_global = blk * 64;
            let mut boundary = (*blocks_state_ptr.add(blk)).boundary;

            if boundary == 0 {
                continue;
            }

            let mut occupied = (*blocks_state_ptr.add(blk)).occupied;
            let initial_occupied = occupied;
            let initial_boundary = boundary;

            let _valid_mask = (*blocks_state_ptr.add(blk)).valid_mask;
            let effective_mask = (*blocks_state_ptr.add(blk)).effective_mask;

            // Horizontal spread only (no intra-block vertical for stride-64)
            let spread_boundary =
                crate::intrinsics::spread_syndrome_linear(boundary, effective_mask);

            // Horizontal union (adjacent bits within same row)
            let horizontal_pairs = spread_boundary & (spread_boundary >> 1) & !ROW_END_MASK;
            if horizontal_pairs != 0 {
                merge_horizontal!(horizontal_pairs, base_global);
            }

            let new_occupied = occupied | spread_boundary;
            if new_occupied != occupied {
                occupied = new_occupied;
                any_expanded = true;
                next_mask |= 1 << blk;
            }

            // UP neighbor (previous block = previous row)
            if blk > 0 {
                let blk_up = blk - 1;
                let valid_up = (*blocks_state_ptr.add(blk_up)).valid_mask;
                let boundary_hits = spread_boundary & !valid_up;
                if boundary_hits != 0 {
                    connect_bits!(boundary_hits, base_global, 0);
                }

                let occupied_up = (*blocks_state_ptr.add(blk_up)).occupied;
                let effective_up = (*blocks_state_ptr.add(blk_up)).effective_mask;
                let grow_up = spread_boundary & !occupied_up & effective_up;
                fast_grow!(blk_up, grow_up);
                if grow_up != 0 {
                    any_expanded = true;
                }

                // Merge with occupied cells in up block
                let merge_up = spread_boundary & occupied_up;
                if merge_up != 0 {
                    let mut m = merge_up;
                    while m != 0 {
                        let bit = tzcnt(m) as usize;
                        m = blsr(m);
                        let u = (base_global + bit) as u32;
                        let v = (blk_up * 64 + bit) as u32;
                        let pu = *parents_ptr.add(u as usize);
                        let pv = *parents_ptr.add(v as usize);
                        if pu != pv {
                            let mut root_u = pu;
                            while root_u != *parents_ptr.add(root_u as usize) {
                                let gp = *parents_ptr.add(root_u as usize);
                                *parents_ptr.add(root_u as usize) = gp;
                                root_u = gp;
                            }
                            let mut root_v = pv;
                            while root_v != *parents_ptr.add(root_v as usize) {
                                let gp = *parents_ptr.add(root_v as usize);
                                *parents_ptr.add(root_v as usize) = gp;
                                root_v = gp;
                            }
                            if root_u != root_v {
                                if root_u < root_v {
                                    *parents_ptr.add(root_u as usize) = root_v;
                                } else {
                                    *parents_ptr.add(root_v as usize) = root_u;
                                }
                                any_expanded = true;
                            }
                        }
                    }
                }
            } else {
                // Block 0: top boundary
                let hits = spread_boundary;
                if hits != 0 {
                    connect_bits!(hits, base_global, 0);
                }
            }

            // DOWN neighbor (next block = next row)
            let blk_down = blk + 1;
            if blk_down < num_blocks {
                let valid_down = (*blocks_state_ptr.add(blk_down)).valid_mask;
                let boundary_hits = spread_boundary & !valid_down;
                if boundary_hits != 0 {
                    connect_bits!(boundary_hits, base_global, 0);
                }

                let occupied_down = (*blocks_state_ptr.add(blk_down)).occupied;
                let effective_down = (*blocks_state_ptr.add(blk_down)).effective_mask;
                let grow_down = spread_boundary & !occupied_down & effective_down;
                fast_grow!(blk_down, grow_down);
                if grow_down != 0 {
                    any_expanded = true;
                }

                // Merge with occupied cells in down block
                let merge_down = spread_boundary & occupied_down;
                if merge_down != 0 {
                    let mut m = merge_down;
                    while m != 0 {
                        let bit = tzcnt(m) as usize;
                        m = blsr(m);
                        let u = (base_global + bit) as u32;
                        let v = (blk_down * 64 + bit) as u32;
                        let pu = *parents_ptr.add(u as usize);
                        let pv = *parents_ptr.add(v as usize);
                        if pu != pv {
                            let mut root_u = pu;
                            while root_u != *parents_ptr.add(root_u as usize) {
                                let gp = *parents_ptr.add(root_u as usize);
                                *parents_ptr.add(root_u as usize) = gp;
                                root_u = gp;
                            }
                            let mut root_v = pv;
                            while root_v != *parents_ptr.add(root_v as usize) {
                                let gp = *parents_ptr.add(root_v as usize);
                                *parents_ptr.add(root_v as usize) = gp;
                                mark_dirty!(root_v as usize >> 6);
                                root_v = gp;
                            }
                            if root_u != root_v {
                                if root_u < root_v {
                                    *parents_ptr.add(root_u as usize) = root_v;
                                } else {
                                    *parents_ptr.add(root_v as usize) = root_u;
                                }
                                any_expanded = true;
                            }
                        }
                    }
                }
            } else {
                // Last block: bottom boundary
                let hits = spread_boundary;
                if hits != 0 {
                    connect_bits!(hits, base_global, 0);
                }
            }

            // Left/Right edge hits
            let left_hits = spread_boundary & ROW_START_MASK;
            if left_hits != 0 {
                connect_bits!(left_hits, base_global, 0);
            }
            let right_hits = spread_boundary & ROW_END_MASK;
            if right_hits != 0 {
                connect_bits!(right_hits, base_global, 0);
            }

            // Update boundary
            boundary &= !spread_boundary;
            if occupied != initial_occupied || boundary != initial_boundary {
                (*blocks_state_ptr.add(blk)).occupied = occupied;
                (*blocks_state_ptr.add(blk)).boundary = boundary;
            }
        }

        (any_expanded, next_mask)
    }

    /// # Safety
    ///
    /// This function is deprecated and always returns false.
    #[inline(always)]
    pub unsafe fn process_block_optimized_32<const SILENT: bool>(
        &mut self,
        _blk_idx: usize,
    ) -> bool {
        // Removed as part of code simplification. Stride 32 now uses small_stride.
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::decoder::state::DecodingState;
    use crate::topology::SquareGrid;

    #[test]
    fn test_optimized_32_unrolled_16() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);
        let boundary_node = (decoder.parents.len() - 1) as u32;
        unsafe {
            let mut syndromes = [0u64; 16];
            syndromes[0] = 1;
            syndromes[1] = 1;
            syndromes[15] = 1u64 << 63;
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = (1 << 0) | (1 << 1) | (1 << 15);
            let (expanded, next_mask) = decoder.process_all_blocks_stride_32_unrolled_16();
            assert!(expanded);
            assert_eq!(next_mask & (1 << 0), 1 << 0);
            assert_eq!(next_mask & (1 << 1), 1 << 1);
            assert_eq!(next_mask & (1 << 15), 1 << 15);
            assert_eq!(decoder.find(0), boundary_node);
            assert_eq!(decoder.find(64), boundary_node);
            assert_eq!(decoder.find(1023), boundary_node);
        }
    }

    /// Test stride_64 processing for 64x64 grids.
    #[test]
    fn test_stride_64_basic() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);
        let boundary_node = (decoder.parents.len() - 1) as u32;

        unsafe {
            // Set syndrome in first block (top-left corner)
            let mut syndromes = std::vec![0u64; decoder.blocks_state.len()];
            syndromes[0] = 1; // bit 0 = node at (0, 0)
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1;

            let (expanded, next_mask) = decoder.process_all_blocks_stride_64();
            assert!(expanded);
            assert!(next_mask != 0);
            assert_eq!(decoder.find(0), boundary_node, "Corner node should connect to boundary");
        }
    }

    /// Test stride_64 with interior nodes (not at boundary).
    #[test]
    fn test_stride_64_interior_nodes() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

        unsafe {
            // Set adjacent syndromes in middle block to trigger horizontal merge
            let mut syndromes = std::vec![0u64; decoder.blocks_state.len()];
            // Block 32 (middle row), nodes at positions 10 and 11 (adjacent)
            syndromes[32] = 0b11 << 10;
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1u64 << 32;

            let (expanded, _next_mask) = decoder.process_all_blocks_stride_64();
            assert!(expanded);

            // Nodes at (10, 32) and (11, 32) should be connected
            let node_a = 32 * 64 + 10;
            let node_b = 32 * 64 + 11;
            assert_eq!(decoder.find(node_a), decoder.find(node_b), "Adjacent nodes should merge");
        }
    }

    /// Test stride_64 with last block (bottom boundary).
    #[test]
    fn test_stride_64_last_block() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);
        let boundary_node = (decoder.parents.len() - 1) as u32;
        let num_blocks = decoder.blocks_state.len();

        unsafe {
            // Set syndrome in last data block
            let mut syndromes = std::vec![0u64; num_blocks];
            let last_data_block = num_blocks - 2; // sentinel is at end
            syndromes[last_data_block] = 1u64 << 63; // bottom-right corner
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1u64 << last_data_block;

            let (expanded, _next_mask) = decoder.process_all_blocks_stride_64();
            assert!(expanded);
            assert_eq!(decoder.find((last_data_block * 64 + 63) as u32), boundary_node);
        }
    }

    /// Test stride_64 with vertical neighbor growth (up).
    #[test]
    fn test_stride_64_vertical_up() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

        unsafe {
            // Place defect in block 5, it should spread to block 4 (up)
            let mut syndromes = std::vec![0u64; decoder.blocks_state.len()];
            syndromes[5] = 1 << 10; // Node in block 5
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1u64 << 5;

            // Run multiple iterations
            for _ in 0..10 {
                let (_, next_mask) = decoder.process_all_blocks_stride_64();
                decoder.active_block_mask = next_mask;
                if next_mask == 0 {
                    break;
                }
            }
            // Just verify it completes without panic
        }
    }

    /// Test stride_64 with vertical neighbor growth (down).
    #[test]
    fn test_stride_64_vertical_down() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

        unsafe {
            // Place defects in block 5 and 6 to test down neighbor merging
            let mut syndromes = std::vec![0u64; decoder.blocks_state.len()];
            syndromes[5] = 1 << 10;
            syndromes[6] = 1 << 10; // Same column, next row
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = (1u64 << 5) | (1u64 << 6);

            // Run iterations
            for _ in 0..20 {
                let (_, next_mask) = decoder.process_all_blocks_stride_64();
                decoder.active_block_mask = next_mask;
                if next_mask == 0 {
                    break;
                }
            }

            // Nodes should be connected via growth
            let node_a = 5 * 64 + 10;
            let node_b = 6 * 64 + 10;
            assert_eq!(decoder.find(node_a as u32), decoder.find(node_b as u32));
        }
    }

    /// Test stride_32 with interior vertical pairs.
    #[test]
    fn test_stride_32_vertical_pairs() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            // Place two defects that form a vertical pair within a block
            // Block layout for stride 32: 2 rows of 32 bits each
            let mut syndromes = [0u64; 17];
            // Bits 0 and 32 are vertically adjacent in stride-32
            syndromes[5] = (1 << 0) | (1 << 32);
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1u64 << 5;

            let (expanded, _) = decoder.process_all_blocks_stride_32_unrolled_16();
            assert!(expanded);

            // The two nodes should be connected
            let node_a = 5 * 64 + 0;
            let node_b = 5 * 64 + 32;
            assert_eq!(decoder.find(node_a as u32), decoder.find(node_b as u32));
        }
    }

    /// Test stride_32 with horizontal pairs.
    #[test]
    fn test_stride_32_horizontal_pairs() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            // Place two horizontally adjacent defects
            let mut syndromes = [0u64; 17];
            syndromes[5] = (1 << 10) | (1 << 11);
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = 1u64 << 5;

            let (expanded, _) = decoder.process_all_blocks_stride_32_unrolled_16();
            assert!(expanded);

            let node_a = 5 * 64 + 10;
            let node_b = 5 * 64 + 11;
            assert_eq!(decoder.find(node_a as u32), decoder.find(node_b as u32));
        }
    }

    /// Test stride_32 inter-block growth (up).
    #[test]
    fn test_stride_32_inter_block_up() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            // Place defects in blocks 7 and 6 (adjacent)
            let mut syndromes = [0u64; 17];
            syndromes[7] = 1 << 0; // Top of block 7
            syndromes[6] = 1u64 << 63; // Bottom of block 6
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = (1u64 << 7) | (1u64 << 6);

            // Run multiple iterations
            for _ in 0..20 {
                let (_, next_mask) = decoder.process_all_blocks_stride_32_unrolled_16();
                decoder.active_block_mask = next_mask;
                if next_mask == 0 {
                    break;
                }
            }

            // Both should connect
            let root_a = decoder.find((7 * 64) as u32);
            let root_b = decoder.find((6 * 64 + 63) as u32);
            assert_eq!(root_a, root_b);
        }
    }

    /// Test stride_32 inter-block growth (down).
    #[test]
    fn test_stride_32_inter_block_down() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            // Place defects in blocks 7 and 8
            let mut syndromes = [0u64; 17];
            syndromes[7] = 1u64 << 63; // Bottom of block 7
            syndromes[8] = 1 << 0; // Top of block 8
            decoder.load_dense_syndromes(&syndromes);
            decoder.active_block_mask = (1u64 << 7) | (1u64 << 8);

            for _ in 0..20 {
                let (_, next_mask) = decoder.process_all_blocks_stride_32_unrolled_16();
                decoder.active_block_mask = next_mask;
                if next_mask == 0 {
                    break;
                }
            }

            let root_a = decoder.find((7 * 64 + 63) as u32);
            let root_b = decoder.find((8 * 64) as u32);
            assert_eq!(root_a, root_b);
        }
    }

    /// Test empty active mask produces no expansion.
    #[test]
    fn test_stride_32_empty_active() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            decoder.active_block_mask = 0;
            let (expanded, next_mask) = decoder.process_all_blocks_stride_32_unrolled_16();
            assert!(!expanded);
            assert_eq!(next_mask, 0);
        }
    }

    /// Test stride_64 empty active mask.
    #[test]
    fn test_stride_64_empty_active() {
        extern crate std;
        let mut memory = std::vec![0u8; 50 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

        unsafe {
            decoder.active_block_mask = 0;
            let (expanded, next_mask) = decoder.process_all_blocks_stride_64();
            assert!(!expanded);
            assert_eq!(next_mask, 0);
        }
    }

    /// Test process_block_optimized_32 (stub function).
    #[test]
    fn test_process_block_optimized_32_stub() {
        extern crate std;
        let mut memory = std::vec![0u8; 10 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        unsafe {
            let result = decoder.process_block_optimized_32::<false>(0);
            assert!(!result);

            let result_silent = decoder.process_block_optimized_32::<true>(0);
            assert!(!result_silent);
        }
    }
}
