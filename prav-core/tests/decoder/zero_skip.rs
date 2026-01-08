use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_zero_skip_loader_patterns() {
    let w = 64;
    let h = 64;
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    let max_dim = w.max(h);
    let pow2 = max_dim.next_power_of_two();
    let stride_y = pow2;
    let total_nodes = stride_y * stride_y;
    let num_blocks = (total_nodes + 63) / 64;

    // Test 1: Empty Input
    {
        decoder.sparse_reset();
        let dense_defects = vec![0u64; num_blocks];
        decoder.load_dense_syndromes(&dense_defects);

        // Verify no blocks touched
        assert_eq!(decoder.blocks_state[0].boundary, 0);
        assert_eq!(decoder.blocks_state[1].boundary, 0);
    }

    // Test 2: Single Bit
    {
        decoder.sparse_reset();
        let mut dense_defects = vec![0u64; num_blocks];
        let blk = 1;
        let bit = 5;
        dense_defects[blk] = 1 << bit;

        decoder.load_dense_syndromes(&dense_defects);

        assert_eq!(decoder.blocks_state[blk].boundary, 1 << bit);
        assert_eq!(decoder.blocks_state[blk].occupied, 1 << bit);
        // Verify other blocks are 0
        assert_eq!(decoder.blocks_state[0].boundary, 0);
        assert_eq!(decoder.blocks_state[2].boundary, 0);
    }

    // Test 3: Multiple bits in same block
    {
        decoder.sparse_reset();
        let mut dense_defects = vec![0u64; num_blocks];
        let blk = 2;
        let pattern = (1 << 0) | (1 << 63) | (1 << 32);
        dense_defects[blk] = pattern;

        decoder.load_dense_syndromes(&dense_defects);

        assert_eq!(decoder.blocks_state[blk].boundary, pattern);
        assert_eq!(decoder.blocks_state[blk].occupied, pattern);
    }

    // Test 4: Multiple blocks
    {
        decoder.sparse_reset();
        let mut dense_defects = vec![0u64; num_blocks];
        dense_defects[0] = 1;
        dense_defects[10] = u64::MAX;
        dense_defects[num_blocks - 2] = 12345;

        decoder.load_dense_syndromes(&dense_defects);

        assert_eq!(decoder.blocks_state[0].boundary, 1);
        assert_eq!(decoder.blocks_state[10].boundary, u64::MAX);
        assert_eq!(decoder.blocks_state[num_blocks - 2].boundary, 12345);
        assert_eq!(decoder.blocks_state[5].boundary, 0); // Intermediate block
    }
}
