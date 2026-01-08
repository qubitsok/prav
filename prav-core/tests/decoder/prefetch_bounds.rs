use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

fn idx(x: usize, y: usize, stride: usize) -> u32 {
    ((y * stride) + x) as u32
}

#[test]
fn test_prefetch_boundary_conditions() {
    let w = 64;
    let h = 64;
    let stride_y = 64; // Power of 2 >= w

    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    // Defects at top boundary
    let d1 = idx(0, 0, stride_y);
    let d2 = idx(10, 0, stride_y);

    // Defects at bottom boundary
    let d3 = idx(0, h - 1, stride_y);
    let d4 = idx(10, h - 1, stride_y);

    let mut corrections = vec![EdgeCorrection::default(); 100];

    let total_nodes = stride_y * stride_y;
    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_syndromes = vec![0u64; num_blocks];

    // Helper to set bit
    let set_bit = |syndromes: &mut [u64], d: u32| {
        let blk = (d as usize) / 64;
        let bit = (d as usize) % 64;
        if blk < syndromes.len() {
            syndromes[blk] |= 1 << bit;
        }
    };

    set_bit(&mut dense_syndromes, d1);
    set_bit(&mut dense_syndromes, d2);
    set_bit(&mut dense_syndromes, d3);
    set_bit(&mut dense_syndromes, d4);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&dense_syndromes);

    // This runs the decoding, which calls process_block_large_stride internally.
    // If prefetch causes OOB or other issues, this might crash.
    let _ = decoder.decode(&mut corrections);
}
