use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_manual_linking_logic_verification() {
    let mut buffer = [0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let u = 10u32;
    let v = 20u32;

    // Simulate what merge_shifted does
    unsafe {
        let pu = *state.parents.get_unchecked(u as usize);
        let pv = *state.parents.get_unchecked(v as usize);

        assert_eq!(pu, u);
        assert_eq!(pv, v);

        if pu == u && pv == v {
            if u < v {
                *state.parents.get_unchecked_mut(u as usize) = v;
                state.mark_block_dirty(u as usize >> 6);
            } else {
                *state.parents.get_unchecked_mut(v as usize) = u;
                state.mark_block_dirty(v as usize >> 6);
            }
        }
    }

    assert_eq!(state.find(u), v);
    // Check if dirty tracking worked (Block 0 is dirty)
    // 10 / 64 = 0.
    assert_ne!(state.block_dirty_mask[0] & (1 << 0), 0);
}

#[test]
fn test_boundary_manual_linking_logic_verification() {
    let mut buffer = [0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut state = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let u = 10u32;
    let boundary_node = (state.parents.len() - 1) as u32;

    // Simulate connect_boundary_shifted
    unsafe {
        let pu = *state.parents.get_unchecked(u as usize);
        if pu == u {
            if u < boundary_node {
                *state.parents.get_unchecked_mut(u as usize) = boundary_node;
                state.mark_block_dirty(u as usize >> 6);
            }
        }
    }

    assert_eq!(state.find(u), boundary_node);
    assert_ne!(state.block_dirty_mask[0] & (1 << 0), 0);
}
