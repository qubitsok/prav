use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_optimistic_root_update_small_stride() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);
    // parents len = 256 + 1 = 257.

    // Setup Block 1.
    unsafe {
        decoder.blocks_state.get_unchecked_mut(0).valid_mask = !0;
        decoder.blocks_state.get_unchecked_mut(2).valid_mask = !0;

        let col_10 = (1 << 10) | (1 << 26) | (1 << 42) | (1 << 58);
        decoder.blocks_state.get_unchecked_mut(1).valid_mask = col_10;

        let block = decoder.blocks_state.get_unchecked_mut(1);
        block.root = 100;
        block.boundary = 1 << 10;
        block.occupied = 1 << 10;
        block.effective_mask = col_10;
    }

    // Case 1: Optimistic Check Passes
    unsafe {
        *decoder.parents.get_unchecked_mut(100) = 100;
    }

    unsafe {
        decoder.process_block_small_stride::<false>(1);
    }

    unsafe {
        let block = decoder.blocks_state.get_unchecked(1);
        assert_eq!(block.root, 100, "Root should remain 100");
    }

    // Case 2: Optimistic Check Fails
    unsafe {
        *decoder.parents.get_unchecked_mut(100) = 200;
        *decoder.parents.get_unchecked_mut(200) = 200;

        let block = decoder.blocks_state.get_unchecked_mut(1);
        block.root = 100;
        // RESET BOUNDARY!
        block.boundary = 1 << 10;
    }

    unsafe {
        decoder.process_block_small_stride::<false>(1);
    }

    unsafe {
        let block = decoder.blocks_state.get_unchecked(1);
        assert_eq!(block.root, 200, "Root should update to 200");
    }

    // Case 3: Optimistic Check Fails (Two hops)
    // parent[200] = 250. parent[250] = 250.
    unsafe {
        *decoder.parents.get_unchecked_mut(200) = 250;
        *decoder.parents.get_unchecked_mut(250) = 250;

        let block = decoder.blocks_state.get_unchecked_mut(1);
        block.root = 100;
        // RESET BOUNDARY!
        block.boundary = 1 << 10;
    }

    unsafe {
        decoder.process_block_small_stride::<false>(1);
    }

    unsafe {
        let block = decoder.blocks_state.get_unchecked(1);
        assert_eq!(block.root, 250, "Root should update to 250");
    }
}

#[test]
fn test_optimistic_root_update_large_stride() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    // Setup Block 1.
    unsafe {
        decoder.blocks_state.get_unchecked_mut(0).valid_mask = !0;
        decoder.blocks_state.get_unchecked_mut(2).valid_mask = !0;

        decoder.blocks_state.get_unchecked_mut(1).valid_mask = 1 << 10;

        let block = decoder.blocks_state.get_unchecked_mut(1);
        block.root = 1000;
        block.boundary = 1 << 10;
        block.occupied = 1 << 10;
        block.effective_mask = 1 << 10;
    }

    // Case 1: Optimistic Check Passes
    unsafe {
        *decoder.parents.get_unchecked_mut(1000) = 1000;
    }

    unsafe {
        decoder.process_block_small_stride::<false>(1);
    }

    unsafe {
        let block = decoder.blocks_state.get_unchecked(1);
        assert_eq!(block.root, 1000);
    }

    // Case 2: Optimistic Check Fails
    unsafe {
        *decoder.parents.get_unchecked_mut(1000) = 2000;
        *decoder.parents.get_unchecked_mut(2000) = 2000;

        let block = decoder.blocks_state.get_unchecked_mut(1);
        block.root = 1000;
        // RESET BOUNDARY
        block.boundary = 1 << 10;
    }

    unsafe {
        decoder.process_block_small_stride::<false>(1);
    }

    unsafe {
        let block = decoder.blocks_state.get_unchecked(1);
        assert_eq!(block.root, 2000);
    }
}
