use crate::decoder::state::DecodingState;
use crate::decoder::union_find::UnionFind;
use crate::topology::Topology;

#[allow(clippy::needless_range_loop)]
impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    #[inline(always)]
    pub(super) unsafe fn fast_grow_block<const SILENT: bool>(
        &mut self,
        blk_idx: usize,
        grow_mask: u64,
    ) {
        if grow_mask == 0 {
            return;
        }

        let block = self.blocks_state.get_unchecked_mut(blk_idx);
        block.occupied |= grow_mask;
        block.boundary |= grow_mask;
        self.mark_block_dirty(blk_idx);

        if !SILENT {
            self.push_next(blk_idx);
        }
    }

    #[inline(always)]
    pub(super) unsafe fn connect_boundary_4way_ilp(
        &mut self,
        mut mask: u64,
        base_u: usize,
        offset_v: isize,
        boundary_node: u32,
    ) -> bool {
        if mask == 0 {
            return false;
        }
        // Optimize for single-bit case (very common)
        if (mask & (mask - 1)) == 0 {
            let bit = tzcnt(mask);
            let u = ((base_u as isize) + offset_v + (bit as isize)) as u32;
            let root_u = self.find(u);
            if self.union_roots(root_u, boundary_node) {
                return true;
            }
            return false;
        }

        let mut expanded = false;
        use crate::intrinsics::{blsr, tzcnt};

        let mut u_indices = [0u32; 4];
        let mut roots_u = [0u32; 4];

        while mask != 0 {
            let mut count = 0;
            for i in 0..4 {
                if mask == 0 {
                    break;
                }
                let bit = tzcnt(mask);
                mask = blsr(mask);

                u_indices[i] = ((base_u as isize) + offset_v + (bit as isize)) as u32;
                count += 1;
            }

            unsafe {
                let parents_ptr = self.parents.as_mut_ptr();

                // Step A: Load initial parents
                for i in 0..count {
                    roots_u[i] = *parents_ptr.add(u_indices[i] as usize);
                }

                // Step B: Path Compression Loop (Interleaved)
                let mut active_lanes = (1 << count) - 1;

                // Pre-check for already connected
                for i in 0..count {
                    if roots_u[i] == boundary_node {
                        active_lanes &= !(1 << i);
                    }
                }

                while active_lanes != 0 {
                    for i in 0..count {
                        if (active_lanes & (1 << i)) != 0 {
                            let p_u = *parents_ptr.add(roots_u[i] as usize);
                            if p_u != roots_u[i] {
                                // Path halving
                                let gp_u = *parents_ptr.add(p_u as usize);
                                *parents_ptr.add(roots_u[i] as usize) = gp_u;
                                roots_u[i] = gp_u;
                            }

                            if roots_u[i] == *parents_ptr.add(roots_u[i] as usize) {
                                active_lanes &= !(1 << i);
                            }
                        }
                    }
                }

                // Step C: Perform Unions
                for i in 0..count {
                    if roots_u[i] != boundary_node {
                        if roots_u[i] < boundary_node {
                            *parents_ptr.add(roots_u[i] as usize) = boundary_node;
                            self.mark_block_dirty(roots_u[i] as usize >> 6);
                        } else {
                            *parents_ptr.add(boundary_node as usize) = roots_u[i];
                            self.mark_block_dirty(boundary_node as usize >> 6);
                        }
                        expanded = true;
                    }
                }
            }
        }
        expanded
    }

    #[inline(always)]
    pub(super) unsafe fn merge_shifted(
        &mut self,
        mask: u64,
        base_src: usize,
        shift: isize,
        base_target: usize,
    ) -> bool {
        let mut expanded = false;
        let mask_lo = mask as u32;
        let mask_hi = (mask >> 32) as u32;

        if mask_lo != 0 && self.merge_shifted_32(mask_lo, base_src, shift, base_target) {
            expanded = true;
        }
        if mask_hi != 0
            && self.merge_shifted_32(mask_hi, base_src + 32, shift, base_target + 32)
        {
            expanded = true;
        }
        expanded
    }

    #[inline(always)]
    unsafe fn merge_shifted_32(
        &mut self,
        mut mask: u32,
        base_src: usize,
        shift: isize,
        base_target: usize,
    ) -> bool {
        let mut expanded = false;
        use crate::intrinsics::tzcnt;

        while mask != 0 {
            let b0 = tzcnt(mask as u64);
            mask &= mask - 1;

            let mut b1 = b0;
            let mut b2 = b0;
            let mut b3 = b0;
            let mut count = 1;

            if mask != 0 {
                b1 = tzcnt(mask as u64);
                mask &= mask - 1;
                count += 1;
            }
            if mask != 0 {
                b2 = tzcnt(mask as u64);
                mask &= mask - 1;
                count += 1;
            }
            if mask != 0 {
                b3 = tzcnt(mask as u64);
                mask &= mask - 1;
                count += 1;
            }

            let u0 = ((base_src + b0 as usize) as isize + shift) as u32;
            let v0 = (base_target + b0 as usize) as u32;
            let u1 = ((base_src + b1 as usize) as isize + shift) as u32;
            let v1 = (base_target + b1 as usize) as u32;
            let u2 = ((base_src + b2 as usize) as isize + shift) as u32;
            let v2 = (base_target + b2 as usize) as u32;
            let u3 = ((base_src + b3 as usize) as isize + shift) as u32;
            let v3 = (base_target + b3 as usize) as u32;

            let p_u0 = *self.parents.get_unchecked(u0 as usize);
            let p_v0 = *self.parents.get_unchecked(v0 as usize);
            let p_u1 = *self.parents.get_unchecked(u1 as usize);
            let p_v1 = *self.parents.get_unchecked(v1 as usize);
            let p_u2 = *self.parents.get_unchecked(u2 as usize);
            let p_v2 = *self.parents.get_unchecked(v2 as usize);
            let p_u3 = *self.parents.get_unchecked(u3 as usize);
            let p_v3 = *self.parents.get_unchecked(v3 as usize);

            // Process 0
            if p_u0 != p_v0 {
                if p_u0 == u0 && p_v0 == v0 {
                    if self.union_roots(u0, v0) {
                        expanded = true;
                    }
                } else if self.union(u0, v0) {
                    expanded = true;
                }
            }

            // Process 1
            if count >= 2 && p_u1 != p_v1 {
                if p_u1 == u1 && p_v1 == v1 {
                    if self.union_roots(u1, v1) {
                        expanded = true;
                    }
                } else if self.union(u1, v1) {
                    expanded = true;
                }
            }

            // Process 2
            if count >= 3 && p_u2 != p_v2 {
                if p_u2 == u2 && p_v2 == v2 {
                    if self.union_roots(u2, v2) {
                        expanded = true;
                    }
                } else if self.union(u2, v2) {
                    expanded = true;
                }
            }

            // Process 3
            if count == 4 && p_u3 != p_v3 {
                if p_u3 == u3 && p_v3 == v3 {
                    if self.union_roots(u3, v3) {
                        expanded = true;
                    }
                } else if self.union(u3, v3) {
                    expanded = true;
                }
            }
        }
        expanded
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::decoder::state::DecodingState;
    use crate::topology::SquareGrid;

    #[test]
    fn test_connect_boundary_4way_ilp_correctness() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Grid 8x8 (64 nodes + boundary)
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let boundary = (decoder.parents.len() - 1) as u32;

        unsafe {
            // Case 1: Simple connection of 4 bits (0, 1, 2, 3) to boundary
            // u = base(0) + bit + offset(0)
            decoder.connect_boundary_4way_ilp(0xF, 0, 0, boundary);

            assert_eq!(decoder.find(0), boundary);
            assert_eq!(decoder.find(1), boundary);
            assert_eq!(decoder.find(2), boundary);
            assert_eq!(decoder.find(3), boundary);

            // Case 2: Offset logic
            // mask = 1 (bit 0). base = 0. offset = 10.
            // u = 0 + 0 + 10 = 10.
            decoder.connect_boundary_4way_ilp(1, 0, 10, boundary);
            assert_eq!(decoder.find(10), boundary);

            // Case 3: Mixed bits and offset
            // mask = 0b101 (bits 0 and 2). base = 20. offset = 5.
            // u1 = 20 + 0 + 5 = 25.
            // u2 = 20 + 2 + 5 = 27.
            decoder.connect_boundary_4way_ilp(0x5, 20, 5, boundary);
            assert_eq!(decoder.find(25), boundary);
            assert_eq!(decoder.find(27), boundary);
        }
    }

    #[test]
    fn test_connect_boundary_fast_path() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let boundary = (decoder.parents.len() - 1) as u32;

        unsafe {
            // Case 1: Single bit connection (should hit fast path)
            // mask = 1 << 5 (bit 5). base = 0. offset = 0. u = 5.
            let res = decoder.connect_boundary_4way_ilp(1 << 5, 0, 0, boundary);
            assert!(res, "Should expand on first connection");
            assert_eq!(decoder.find(5), boundary);

            // Case 2: Already connected (should return false)
            let res = decoder.connect_boundary_4way_ilp(1 << 5, 0, 0, boundary);
            assert!(!res, "Should not expand if already connected");

            // Case 3: Single bit with offset (should hit fast path)
            // mask = 1 << 0. base = 10. offset = 5. u = 10 + 0 + 5 = 15.
            let res = decoder.connect_boundary_4way_ilp(1, 10, 5, boundary);
            assert!(res);
            assert_eq!(decoder.find(15), boundary);
        }
    }

    #[test]
    fn test_merge_shifted_runs() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // mask = 0b1110 (bits 1, 2, 3). Run length 3.
            // base_src = 10. shift = 0. base_target = 20.
            // Pairs: (11, 21), (12, 22), (13, 23).
            let res = decoder.merge_shifted(0b1110, 10, 0, 20);
            assert!(res);

            assert_eq!(decoder.find(11), decoder.find(21));
            assert_eq!(decoder.find(12), decoder.find(22));
            assert_eq!(decoder.find(13), decoder.find(23));

            // Should NOT connect bit 0 (index 10 and 20)
            assert_ne!(decoder.find(10), decoder.find(20));

            // Test disjoint runs
            // mask = 0b10010001 (bits 0, 4, 7).
            // shift = 30.
            // base_src = 0. base_target = 0.
            // Pairs: (30, 0), (34, 4), (37, 7).
            let res = decoder.merge_shifted(0b10010001, 0, 30, 0);
            assert!(res);

            assert_eq!(decoder.find(30), decoder.find(0));
            assert_eq!(decoder.find(34), decoder.find(4));
            assert_eq!(decoder.find(37), decoder.find(7));
        }
    }

    /// Test connect_boundary_4way_ilp with empty mask (early return).
    #[test]
    fn test_connect_boundary_empty_mask() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let boundary = (decoder.parents.len() - 1) as u32;

        unsafe {
            // Empty mask should return false immediately
            let res = decoder.connect_boundary_4way_ilp(0, 0, 0, boundary);
            assert!(!res, "Empty mask should return false");
        }
    }

    /// Test connect_boundary_4way_ilp with many bits (>4, multiple batches).
    #[test]
    fn test_connect_boundary_multi_batch() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        let boundary = (decoder.parents.len() - 1) as u32;

        unsafe {
            // 8 bits set - requires 2 batches of 4
            let res = decoder.connect_boundary_4way_ilp(0xFF, 0, 0, boundary);
            assert!(res);

            for i in 0..8 {
                assert_eq!(decoder.find(i), boundary, "Node {} should be connected", i);
            }
        }
    }

    /// Test merge_shifted with high 32 bits (bits 32-63).
    #[test]
    fn test_merge_shifted_high_bits() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024 * 10];
        let mut arena = Arena::new(&mut memory);
        // Need larger grid for high bit indices
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Mask with bits in the high 32 bits (32-63)
            // bit 32: pair (base_src + 32, base_target + 32)
            let mask: u64 = 1 << 32;
            let res = decoder.merge_shifted(mask, 0, 8, 0);
            assert!(res);

            // u = 32 + 8 = 40, v = 32
            assert_eq!(decoder.find(40), decoder.find(32));
        }
    }

    /// Test merge_shifted with both low and high bits.
    #[test]
    fn test_merge_shifted_both_halves() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024 * 10];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

        unsafe {
            // Bits in both halves: bit 0 (low) and bit 32 (high)
            let mask: u64 = (1 << 0) | (1 << 32);
            let res = decoder.merge_shifted(mask, 0, 16, 0);
            assert!(res);

            // Low half: u = 0 + 16 = 16, v = 0
            assert_eq!(decoder.find(16), decoder.find(0));

            // High half: u = 32 + 16 = 48, v = 32
            assert_eq!(decoder.find(48), decoder.find(32));
        }
    }

    /// Test merge_shifted with exactly 2 bits (count=2 branch).
    #[test]
    fn test_merge_shifted_two_bits() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // 2 bits: bits 0 and 1
            let res = decoder.merge_shifted(0b11, 10, 8, 10);
            assert!(res);

            // (10+8, 10) = (18, 10), (11+8, 11) = (19, 11)
            assert_eq!(decoder.find(18), decoder.find(10));
            assert_eq!(decoder.find(19), decoder.find(11));
        }
    }

    /// Test merge_shifted with exactly 4 bits (full batch).
    #[test]
    fn test_merge_shifted_four_bits() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // 4 bits: bits 0, 1, 2, 3
            let res = decoder.merge_shifted(0xF, 20, 8, 20);
            assert!(res);

            // Pairs: (28, 20), (29, 21), (30, 22), (31, 23)
            assert_eq!(decoder.find(28), decoder.find(20));
            assert_eq!(decoder.find(29), decoder.find(21));
            assert_eq!(decoder.find(30), decoder.find(22));
            assert_eq!(decoder.find(31), decoder.find(23));
        }
    }

    /// Test merge_shifted when nodes are already connected (no expansion).
    #[test]
    fn test_merge_shifted_already_connected() {
        extern crate std;
        let mut memory = std::vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe {
            // First merge
            let res1 = decoder.merge_shifted(0b1, 10, 8, 10);
            assert!(res1);

            // Same merge again - should return false (already connected)
            let res2 = decoder.merge_shifted(0b1, 10, 8, 10);
            assert!(!res2, "Should return false when already connected");
        }
    }
}
