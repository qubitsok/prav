use prav_core::arena::Arena;
use prav_core::decoder::growth::ClusterGrowth;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::Topology;

#[derive(Clone, Copy)]
struct SquareGrid;
impl Topology for SquareGrid {
    fn for_each_neighbor<F>(_node: u32, _f: F)
    where
        F: FnMut(u32),
    {
    }
}

#[test]
fn test_swar_east_expansion() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    // 8x8 grid ensures stride_y = 8, triggering optimized_8x8
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Set defect at (0, 0) -> Index 0
    let syndromes = [1u64]; // Bit 0 set
    state.load_dense_syndromes(&syndromes);

    // Grow
    state.grow_iteration();

    // Expect growth to (1, 0) -> Index 1
    // Parent of 1 should be 0
    assert_eq!(state.find(1), state.find(0));
    // Check occupied
    let occupied = state.blocks_state[0].occupied;
    assert_eq!(occupied & (1 << 1), 1 << 1, "Bit 1 should be occupied");
}

#[test]
fn test_swar_west_expansion() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Set defect at (1, 0) -> Index 1
    let syndromes = [1u64 << 1];
    state.load_dense_syndromes(&syndromes);

    state.grow_iteration();

    // Expect growth to (0, 0) -> Index 0
    assert_eq!(state.find(0), state.find(1));
    let occupied = state.blocks_state[0].occupied;
    assert_eq!(occupied & 1, 1, "Bit 0 should be occupied");
}

#[test]
fn test_swar_south_expansion() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Set defect at (0, 0) -> Index 0
    let syndromes = [1u64];
    state.load_dense_syndromes(&syndromes);

    state.grow_iteration();

    // Expect growth to (0, 1) -> Index 8 (stride=8)
    assert_eq!(state.find(8), state.find(0));
    let occupied = state.blocks_state[0].occupied;
    assert_eq!(occupied & (1 << 8), 1 << 8, "Bit 8 should be occupied");
}

#[test]
fn test_swar_north_expansion() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Set defect at (0, 1) -> Index 8
    let syndromes = [1u64 << 8];
    state.load_dense_syndromes(&syndromes);

    state.grow_iteration();

    // Expect growth to (0, 0) -> Index 0
    assert_eq!(state.find(0), state.find(8));
    let occupied = state.blocks_state[0].occupied;
    assert_eq!(occupied & 1, 1, "Bit 0 should be occupied");
}

#[test]
fn test_swar_collision_merge() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Defect at (0, 0) and (2, 0) -> Index 0 and 2
    let syndromes = [1u64 | (1u64 << 2)];
    state.load_dense_syndromes(&syndromes);

    state.grow_iteration();

    // (0, 0) grows East to (1, 0).
    // (2, 0) grows West to (1, 0).
    // Should merge at (1, 0).

    assert_eq!(state.find(0), state.find(2));
    assert_eq!(state.find(1), state.find(0));
}

#[test]
fn test_swar_full_cross_growth() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Defect at (1, 1) -> Index 9
    let syndromes = [1u64 << 9];
    state.load_dense_syndromes(&syndromes);

    state.grow_iteration();

    // Expect growth in all 4 directions:
    // East: (2, 1) -> 10
    // West: (0, 1) -> 8
    // South: (1, 2) -> 17
    // North: (1, 0) -> 1

    assert_eq!(state.find(10), state.find(9));
    assert_eq!(state.find(8), state.find(9));
    assert_eq!(state.find(17), state.find(9));
    assert_eq!(state.find(1), state.find(9));
}
