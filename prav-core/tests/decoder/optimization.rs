use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::decoder::union_find::UnionFind;
use prav_core::topology::SquareGrid;

#[test]
fn test_union_roots_selective_invalidation() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // Grid 64x64 (Stride 64). 4096 nodes. 64 blocks.
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    // Pick two nodes in different blocks
    let node_u = 0u32; // Block 0
    let node_v = 64u32; // Block 1

    unsafe {
        // Initialize roots
        // Set parents
        *decoder.parents.get_unchecked_mut(node_u as usize) = node_u;
        *decoder.parents.get_unchecked_mut(node_v as usize) = node_v;

        // Set block state roots
        decoder.blocks_state.get_unchecked_mut(0).root = node_u;
        decoder.blocks_state.get_unchecked_mut(1).root = node_v;

        // Perform union: node_u (0) < node_v (64)
        // union_roots(0, 64) -> 0 joins 64 (parent[0] = 64)
        // Block 0 cache should be invalidated (set to MAX).
        // Block 1 cache should remain valid (set to 64).

        let result = decoder.union_roots(node_u, node_v);
        assert!(result);

        // Check parents
        assert_eq!(*decoder.parents.get_unchecked(node_u as usize), node_v);
        assert_eq!(*decoder.parents.get_unchecked(node_v as usize), node_v);

        // Check cache invalidation
        let root_cache_0 = decoder.blocks_state.get_unchecked(0).root;
        let root_cache_1 = decoder.blocks_state.get_unchecked(1).root;

        assert_eq!(
            root_cache_0,
            u32::MAX,
            "Block 0 cache should be invalidated (u joins v)"
        );
        assert_eq!(root_cache_1, node_v, "Block 1 cache should remain valid");
    }
}

#[test]
fn test_union_roots_selective_invalidation_reverse() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    let node_u = 64u32; // Block 1
    let node_v = 0u32; // Block 0

    unsafe {
        *decoder.parents.get_unchecked_mut(node_u as usize) = node_u;
        *decoder.parents.get_unchecked_mut(node_v as usize) = node_v;

        decoder.blocks_state.get_unchecked_mut(1).root = node_u;
        decoder.blocks_state.get_unchecked_mut(0).root = node_v;

        let result = decoder.union_roots(node_u, node_v);
        assert!(result);

        assert_eq!(*decoder.parents.get_unchecked(node_v as usize), node_u);

        let root_cache_0 = decoder.blocks_state.get_unchecked(0).root;
        let root_cache_1 = decoder.blocks_state.get_unchecked(1).root;

        assert_eq!(
            root_cache_0,
            u32::MAX,
            "Block 0 cache should be invalidated (v joins u)"
        );
        assert_eq!(root_cache_1, node_u, "Block 1 cache should remain valid");
    }
}
