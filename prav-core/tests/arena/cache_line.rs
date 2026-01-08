use prav_core::arena::Arena;
use std::mem::align_of;

#[test]
fn test_arena_alignment_u16_64byte() {
    let mut memory = vec![0u8; 1024];
    let mut arena = Arena::new(&mut memory);

    // Alloc u16 slice aligned to 64 bytes
    let slice = arena.alloc_slice_aligned::<u16>(32, 64).unwrap();
    let ptr = slice.as_ptr() as usize;

    assert_eq!(ptr % 64, 0, "Pointer should be 64-byte aligned");
    assert_eq!(align_of::<u16>(), 2);
}

#[test]
fn test_stripe_partitioning_logic() {
    // Replicating get_stripe_range logic for verification
    fn get_stripe_range(t_idx: usize, num_threads: usize, total_blocks: usize) -> (usize, usize) {
        let align = 32;
        let blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
        let aligned_per_thread = ((blocks_per_thread + align - 1) / align) * align;

        let start = (t_idx * aligned_per_thread).min(total_blocks);
        let end = (start + aligned_per_thread).min(total_blocks);
        (start, end)
    }

    // Case B: 16 blocks (32x32), 4 threads
    // Should result in Thread 0 taking all, others empty.
    let (s0, e0) = get_stripe_range(0, 4, 16);
    let (s1, e1) = get_stripe_range(1, 4, 16);

    assert_eq!(s0, 0);
    assert_eq!(e0, 16); // 16 < 32 (aligned size)
    assert_eq!(s1, 16);
    assert_eq!(e1, 16);

    // Case A: 1600 blocks, 4 threads
    // Aligned per thread = 416
    let (s0, e0) = get_stripe_range(0, 4, 1600);
    let (s1, e1) = get_stripe_range(1, 4, 1600);

    assert_eq!(s0, 0);
    assert_eq!(e0, 416);
    assert_eq!(s1, 416);
    assert_eq!(e1, 832);

    // Boundary check: End of last thread
    let (s3, e3) = get_stripe_range(3, 4, 1600);
    assert_eq!(s3, 1248);
    assert_eq!(e3, 1600);
}
