use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_optimized_32_signature_and_logic() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // 32x32 grid -> Stride 32.
    // 1024 nodes. 16 blocks (64 nodes per block).
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Setup state manually to test process_block_small_stride_32
    // We will test block 0.
    let blk_idx = 0;

    // 1. Monochromatic Growth
    unsafe {
        let block = state.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = 1; // Bit 0 active
        block.occupied = 1;
    }

    unsafe {
        state.process_block_small_stride::<false>(blk_idx);
    }

    // Verify spread
    let occupied = unsafe { state.blocks_state.get_unchecked(blk_idx).occupied };
    assert_ne!(occupied & 2, 0, "Bit 0 should spread to Bit 1");

    let root0 = state.find(0);
    let root1 = state.find(1);
    assert_eq!(root0, root1);

    // 2. Polychromatic Merge
    // Reset state for block 1
    let blk_idx = 1;
    unsafe {
        let block = state.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = (1 << 0) | (1 << 2);
        block.occupied = (1 << 0) | (1 << 2);

        // Make them have different roots
        let base = blk_idx * 64;
        state.parents[base + 0] = (base + 0) as u32;
        state.parents[base + 2] = (base + 2) as u32;
    }

    unsafe {
        state.process_block_small_stride::<false>(blk_idx);
    }

    // They should merge at bit 1 (0 -> 1 <- 2)
    let root0 = state.find((blk_idx * 64 + 0) as u32);
    let root2 = state.find((blk_idx * 64 + 2) as u32);
    assert_eq!(
        root0, root2,
        "Disjoint trees should merge via common neighbor"
    );
}

#[test]
fn test_optimized_32_inter_block() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Block 0 covers Row 0 (0..31) and Row 1 (32..63).
    // Block 1 covers Row 2 and Row 3.

    // Let's set a bit at index 63 (Block 0, last bit).
    // It should grow down to index 64 (Block 1, first bit).
    // Down neighbor of 63 is 63 + 32 = 95.
    // 95 is in Block 1. 95 % 64 = 31.

    unsafe {
        let blk0 = 0;
        let block0 = state.blocks_state.get_unchecked_mut(blk0);
        block0.boundary = 1 << 63;
        block0.occupied = 1 << 63;
    }

    unsafe {
        state.process_block_small_stride::<false>(0);
    }

    // Check if Block 1 was activated or grown into
    let occupied1 = unsafe { state.blocks_state.get_unchecked(1).occupied };
    // 63 -> Down is 95. 95 is bit 31 in Block 1.
    assert_eq!(
        occupied1 & (1 << 31),
        1 << 31,
        "Growth should cross block boundary downwards"
    );

    let root63 = state.find(63);
    let root95 = state.find(95);
    assert_eq!(root63, root95);
}
