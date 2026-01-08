use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;
extern crate alloc;
use alloc::vec;

#[test]
fn test_monochromatic_unrolling_large_stride() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // Stride 64 -> Large Stride
    let mut state = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    let root = 0;

    unsafe {
        // Setup block 0 as monochromatic
        state.blocks_state.get_unchecked_mut(0).root = root;

        // We want multiple bits in spread_boundary.
        // bit 10 spreads to 9, 11.
        // bit 15 spreads to 14, 16.
        // bit 20 spreads to 19, 21.
        // bit 30 spreads to 29, 31.
        // bit 40 spreads to 39, 41.
        // Total 5 bits in boundary -> 15 bits in spread_boundary (approx).
        // This ensures the unrolled loop (stride 4) executes multiple times + remainder.
        let boundary_mask = (1 << 10) | (1 << 15) | (1 << 20) | (1 << 30) | (1 << 40);

        let block = state.blocks_state.get_unchecked_mut(0);
        block.boundary = boundary_mask;
        block.occupied = 0;

        let block_static = state.blocks_state.get_unchecked_mut(0);
        block_static.valid_mask = !0;
        block_static.erasure_mask = 0;

        // Ensure parents are NOT root initially
        for i in 0..64 {
            *state.parents.get_unchecked_mut(i) = 999;
        }
        // Fix: Root must point to itself
        *state.parents.get_unchecked_mut(root as usize) = root;

        state.process_block_small_stride::<false>(0);

        // Check if parents are set correctly.
        // spread_syndrome_linear(boundary, mask)
        // linear spread: bit i -> i-1, i, i+1 (if within bounds and mask allows)
        // We expect neighbors of the boundary bits to point to root (0).

        let expected_set_bits = [9, 10, 11, 14, 15, 16, 19, 20, 21, 29, 30, 31, 39, 40, 41];

        for &bit in &expected_set_bits {
            let p = *state.parents.get_unchecked(bit);
            assert_eq!(
                p, root,
                "Node {} should have parent {}, got {}",
                bit, root, p
            );
        }
    }
}

#[test]
fn test_monochromatic_unrolling_small_stride() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // Stride 16 -> Small Stride
    let mut state = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let root = 0;

    unsafe {
        // Setup block 0 as monochromatic
        state.blocks_state.get_unchecked_mut(0).root = root;

        // Boundary bits.
        // Small stride spread includes vertical spread too?
        // In process_block_small_stride:
        // Task 1: Linear spread (Task 1.5 adds vertical)

        let boundary_mask = (1 << 5) | (1 << 8) | (1 << 12) | (1 << 20) | (1 << 25);

        let block = state.blocks_state.get_unchecked_mut(0);
        block.boundary = boundary_mask;
        block.occupied = 0;

        let block_static = state.blocks_state.get_unchecked_mut(0);
        block_static.valid_mask = !0;
        block_static.erasure_mask = 0;

        for i in 0..64 {
            *state.parents.get_unchecked_mut(i) = 999;
        }
        *state.parents.get_unchecked_mut(root as usize) = root;

        state.process_block_small_stride::<false>(0);

        // We just verify that *some* nodes got updated to root.
        // Specifically the boundary bits themselves and their neighbors should be updated.
        // Since occupied was 0, they are "newly_occupied".

        let check_bits = [5, 8, 12, 20, 25];
        for &bit in &check_bits {
            let p = *state.parents.get_unchecked(bit);
            assert_eq!(
                p, root,
                "Node {} (boundary) should have parent {}, got {}",
                bit, root, p
            );
        }

        // Check a neighbor (linear)
        let neighbor = 6; // 5 -> 6
        let p = *state.parents.get_unchecked(neighbor);
        assert_eq!(
            p, root,
            "Node {} (neighbor) should have parent {}",
            neighbor, root
        );
    }
}
