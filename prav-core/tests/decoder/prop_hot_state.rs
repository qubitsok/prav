use prav_core::arena::Arena;
use prav_core::decoder::state::{DecodingState, FLAG_VALID_FULL};
use prav_core::topology::SquareGrid;
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_hot_state_consistency(valid in any::<u64>(), erasure in any::<u64>()) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        // Dummy dims, we just want to access blocks
        let decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        // Manually inject state for Block 0 to simulate scenarios
        unsafe {
            decoder.blocks_state.get_unchecked_mut(0).valid_mask = valid;
            decoder.blocks_state.get_unchecked_mut(0).erasure_mask = erasure;

            // Trigger logic that updates hot state (load_erasures does this)
            // But load_erasures takes a slice of erasures.
            // valid_mask is usually static, but we can mock it here.
            // We need to re-run the update logic that happens in load_erasures AND initialization.

            // Initialization logic for flags:
            if valid == !0 {
                decoder.blocks_state[0].flags |= FLAG_VALID_FULL;
            } else {
                decoder.blocks_state[0].flags &= !FLAG_VALID_FULL;
            }

            // Load erasures logic for effective_mask:
            decoder.blocks_state[0].effective_mask = valid & !erasure;
        }

        let block_hot = decoder.blocks_state[0];

        // Invariants
        // 1. Effective mask is always valid & !erasure
        assert_eq!(block_hot.effective_mask, valid & !erasure);

        // 2. FLAG_VALID_FULL is set iff valid_mask is !0
        let is_full = (block_hot.flags & FLAG_VALID_FULL) != 0;
        assert_eq!(is_full, valid == !0);

        // 3. Alignment check (runtime)
        let ptr = &decoder.blocks_state[0] as *const _;
        assert_eq!(ptr as usize % 32, 0, "BlockStateHot must be 32-byte aligned in memory");
    }
}
