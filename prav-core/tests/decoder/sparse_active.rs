use prav_core::arena::Arena;
use prav_core::decoder::growth::ClusterGrowth;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::Topology;

#[derive(Clone, Copy)]
struct MockTopology;
impl Topology for MockTopology {
    fn for_each_neighbor<F>(_u: u32, _f: F)
    where
        F: FnMut(u32),
    {
    }
}

#[test]
fn test_sparse_active_propagation() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let width = 16;
    let height = 16;

    // Create state
    let mut state = DecodingState::<MockTopology, 16>::new(&mut arena, width, height, 1);

    // Inject a syndrome in block 0
    let mut syndromes = vec![0u64; state.blocks_state.len()];
    syndromes[0] = 1; // Bit 0 in block 0

    state.load_dense_syndromes(&syndromes);

    // Check initial active set
    if !state.is_small_grid() {
        assert_ne!(state.active_mask[0] & 1, 0, "Block 0 should be active");
    } else {
        // Small grid optimization: active_block_mask is used instead of queued_mask
        assert_ne!(state.active_block_mask, 0);
    }

    // Run one iteration manually
    let expanded = state.grow_iteration();
    assert!(expanded);

    // Let's verify correctness with a full grow
    state.initialize_internal();
    state.load_dense_syndromes(&syndromes);
    state.grow_clusters();

    // If it didn't crash and finished, that's a good sign.
    assert_ne!(state.blocks_state[0].occupied, 0);
}
