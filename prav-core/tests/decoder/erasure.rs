use prav_core::{
    arena::Arena, decoder::EdgeCorrection, qec_engine::QecEngine, topology::SquareGrid,
};

#[test]
fn test_erasure_handling() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 4x4 Grid
    // 0  1  2  3
    // 4  5  6  7
    // ...
    let mut engine = QecEngine::<SquareGrid, 4>::new(&mut arena, 4, 4, 1);

    // Set up erasures: Disconnect node 1 from 0 (and everything else)
    // Erasing node 1.
    engine.load_erasures(&[1u64 << 1]); // This method now exists

    // Defects at 0 and 2.
    // If 1 is erased, 0 cannot reach 2 via 1.
    // In a 4x4 grid, 0 is (0,0), 1 is (1,0), 2 is (2,0).
    // Neighbors of 0: 1, 4. Neighbors of 1: 0, 2, 5. Neighbors of 2: 1, 3, 6.
    // If 1 is erased:
    // 0 has neighbor 4.
    // 2 has neighbors 3, 6.

    // We expect them to eventually match, but not through 1.
    // Let's create defects.
    let mut defects = vec![0u64; 1];
    defects[0] = (1 << 0) | (1 << 2);

    let mut corrections = vec![EdgeCorrection::default(); 100];
    let count = engine.process_cycle_dense(&defects, &mut corrections);

    assert!(count > 0);

    // Verify that NO correction involves node 1.
    for i in 0..count {
        let c = corrections[i];
        assert_ne!(c.u, 1, "Correction involved erased node 1");
        assert_ne!(c.v, 1, "Correction involved erased node 1");
    }
}
