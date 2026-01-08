//! Unit tests for decoder types: EdgeCorrection, BlockStateHot, BoundaryConfig.
//!
//! These tests verify the core data structures used in the decoder:
//! - Default implementations produce expected values
//! - Memory layout and alignment are correct for cache efficiency
//! - Constants have expected values

use prav_core::decoder::types::{BlockStateHot, BoundaryConfig, EdgeCorrection, FLAG_VALID_FULL};

// =============================================================================
// EdgeCorrection Tests
// =============================================================================

#[test]
fn test_edge_correction_default() {
    let ec = EdgeCorrection::default();
    assert_eq!(ec.u, 0, "EdgeCorrection::default().u should be 0");
    assert_eq!(ec.v, 0, "EdgeCorrection::default().v should be 0");
}

#[test]
fn test_edge_correction_boundary_sentinel() {
    // Boundary corrections use v == u32::MAX as sentinel
    let boundary_correction = EdgeCorrection { u: 42, v: u32::MAX };
    assert_eq!(boundary_correction.v, u32::MAX, "Boundary corrections use u32::MAX as v");
}

#[test]
fn test_edge_correction_ordering() {
    // EdgeCorrection implements PartialOrd/Ord for sorting
    let a = EdgeCorrection { u: 1, v: 2 };
    let b = EdgeCorrection { u: 1, v: 3 };
    let c = EdgeCorrection { u: 2, v: 1 };

    assert!(a < b, "Same u, smaller v should be less");
    assert!(a < c, "Smaller u should be less");
    assert!(b < c, "Smaller u should be less regardless of v");
}

#[test]
fn test_edge_correction_equality() {
    let a = EdgeCorrection { u: 10, v: 20 };
    let b = EdgeCorrection { u: 10, v: 20 };
    let c = EdgeCorrection { u: 10, v: 21 };

    assert_eq!(a, b, "Identical EdgeCorrections should be equal");
    assert_ne!(a, c, "Different EdgeCorrections should not be equal");
}

#[test]
fn test_edge_correction_copy() {
    let a = EdgeCorrection { u: 5, v: 10 };
    let b = a; // Copy
    assert_eq!(a.u, b.u, "Copy should preserve u");
    assert_eq!(a.v, b.v, "Copy should preserve v");
}

// =============================================================================
// BlockStateHot Tests
// =============================================================================

#[test]
fn test_block_state_hot_default() {
    let block = BlockStateHot::default();

    assert_eq!(block.boundary, 0, "boundary should be 0");
    assert_eq!(block.occupied, 0, "occupied should be 0");
    assert_eq!(block.effective_mask, 0, "effective_mask should be 0");
    assert_eq!(block.valid_mask, 0, "valid_mask should be 0");
    assert_eq!(block.erasure_mask, 0, "erasure_mask should be 0");
    assert_eq!(block.root, u32::MAX, "root should be u32::MAX (invalid)");
    assert_eq!(block.flags, 0, "flags should be 0");
    assert_eq!(block.root_rank, 0, "root_rank should be 0");
    assert_eq!(block._reserved, [0; 7], "reserved should be zeroed");
    assert_eq!(block._padding, [0; 8], "padding should be zeroed");
}

#[test]
fn test_block_state_hot_alignment() {
    // BlockStateHot should be 64-byte aligned for cache line optimization
    let alignment = core::mem::align_of::<BlockStateHot>();
    assert_eq!(alignment, 64, "BlockStateHot must be 64-byte aligned for cache line optimization");
}

#[test]
fn test_block_state_hot_size() {
    // BlockStateHot should be exactly 64 bytes (one cache line)
    let size = core::mem::size_of::<BlockStateHot>();
    assert_eq!(size, 64, "BlockStateHot must be exactly 64 bytes (one cache line)");
}

#[test]
fn test_block_state_hot_field_offsets() {
    // Verify the struct layout is as expected for cache efficiency
    // All u64 fields should be naturally aligned (8-byte boundaries)
    use core::mem::offset_of;

    // boundary, occupied, effective_mask, valid_mask, erasure_mask are u64 (8 bytes each)
    // root is u32 (4 bytes), flags is u32 (4 bytes)
    // root_rank is u8 (1 byte), _reserved is [u8; 7], _padding is [u8; 8]
    // Total: 5*8 + 4 + 4 + 1 + 7 + 8 = 64 bytes

    assert_eq!(offset_of!(BlockStateHot, boundary), 0);
    assert_eq!(offset_of!(BlockStateHot, occupied), 8);
    assert_eq!(offset_of!(BlockStateHot, effective_mask), 16);
    assert_eq!(offset_of!(BlockStateHot, valid_mask), 24);
    assert_eq!(offset_of!(BlockStateHot, erasure_mask), 32);
    assert_eq!(offset_of!(BlockStateHot, root), 40);
    assert_eq!(offset_of!(BlockStateHot, flags), 44);
    assert_eq!(offset_of!(BlockStateHot, root_rank), 48);
    assert_eq!(offset_of!(BlockStateHot, _reserved), 49);
    assert_eq!(offset_of!(BlockStateHot, _padding), 56);
}

#[test]
fn test_block_state_hot_copy() {
    let mut original = BlockStateHot::default();
    original.boundary = 0xDEAD_BEEF_CAFE_BABE;
    original.root = 42;
    original.flags = FLAG_VALID_FULL;

    let copied = original; // Copy

    assert_eq!(copied.boundary, original.boundary, "Copy should preserve boundary");
    assert_eq!(copied.root, original.root, "Copy should preserve root");
    assert_eq!(copied.flags, original.flags, "Copy should preserve flags");
}

#[test]
fn test_block_state_hot_root_invalid_sentinel() {
    // u32::MAX is used as "invalid" sentinel for cached root
    let block = BlockStateHot::default();
    assert_eq!(block.root, u32::MAX, "Default root should be u32::MAX (invalid sentinel)");

    // After a union operation, root might be cached
    let mut block_with_cached = BlockStateHot::default();
    block_with_cached.root = 123; // Simulate cached root
    assert_ne!(block_with_cached.root, u32::MAX, "Cached root should not be u32::MAX");
}

// =============================================================================
// BoundaryConfig Tests
// =============================================================================

#[test]
fn test_boundary_config_default() {
    let config = BoundaryConfig::default();

    assert!(config.check_top, "check_top should default to true");
    assert!(config.check_bottom, "check_bottom should default to true");
    assert!(config.check_left, "check_left should default to true");
    assert!(config.check_right, "check_right should default to true");
}

#[test]
fn test_boundary_config_partial_boundaries() {
    // Test creating configs with partial boundaries (e.g., for tiled grids)
    let config = BoundaryConfig {
        check_top: true,
        check_bottom: false,
        check_left: false,
        check_right: true,
    };

    assert!(config.check_top, "check_top should be true");
    assert!(!config.check_bottom, "check_bottom should be false");
    assert!(!config.check_left, "check_left should be false");
    assert!(config.check_right, "check_right should be true");
}

#[test]
fn test_boundary_config_no_boundaries() {
    // Interior tile with no physical boundaries
    let config = BoundaryConfig {
        check_top: false,
        check_bottom: false,
        check_left: false,
        check_right: false,
    };

    assert!(!config.check_top);
    assert!(!config.check_bottom);
    assert!(!config.check_left);
    assert!(!config.check_right);
}

#[test]
fn test_boundary_config_copy() {
    let original = BoundaryConfig {
        check_top: true,
        check_bottom: false,
        check_left: true,
        check_right: false,
    };

    let copied = original; // Copy

    assert_eq!(copied.check_top, original.check_top);
    assert_eq!(copied.check_bottom, original.check_bottom);
    assert_eq!(copied.check_left, original.check_left);
    assert_eq!(copied.check_right, original.check_right);
}

// =============================================================================
// FLAG_VALID_FULL Tests
// =============================================================================

#[test]
fn test_flag_valid_full_value() {
    // FLAG_VALID_FULL should be bit 0
    assert_eq!(FLAG_VALID_FULL, 1, "FLAG_VALID_FULL should be 1 (bit 0)");
}

#[test]
fn test_flag_valid_full_usage() {
    let mut block = BlockStateHot::default();

    // Initially no flags set
    assert_eq!(block.flags & FLAG_VALID_FULL, 0, "FLAG_VALID_FULL should not be set initially");

    // Set the flag
    block.flags |= FLAG_VALID_FULL;
    assert_eq!(block.flags & FLAG_VALID_FULL, FLAG_VALID_FULL, "FLAG_VALID_FULL should be set");

    // Clear the flag
    block.flags &= !FLAG_VALID_FULL;
    assert_eq!(block.flags & FLAG_VALID_FULL, 0, "FLAG_VALID_FULL should be cleared");
}

#[test]
fn test_flag_valid_full_with_other_flags() {
    // FLAG_VALID_FULL should be independent of other potential flags
    let mut block = BlockStateHot::default();

    // Set FLAG_VALID_FULL and some hypothetical other bits
    block.flags = FLAG_VALID_FULL | 0b1010_0000; // Set bit 0 and some high bits

    // FLAG_VALID_FULL should still be detectable
    assert_ne!(block.flags & FLAG_VALID_FULL, 0, "FLAG_VALID_FULL should be detectable with other flags");

    // Clear only FLAG_VALID_FULL
    block.flags &= !FLAG_VALID_FULL;
    assert_eq!(block.flags & FLAG_VALID_FULL, 0, "FLAG_VALID_FULL should be cleared");
    assert_eq!(block.flags, 0b1010_0000, "Other flags should remain");
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

#[test]
fn test_block_state_hot_array_alignment() {
    // When allocated in an array, each BlockStateHot should maintain alignment
    let blocks: [BlockStateHot; 4] = [
        BlockStateHot::default(),
        BlockStateHot::default(),
        BlockStateHot::default(),
        BlockStateHot::default(),
    ];

    let base_addr = &blocks[0] as *const _ as usize;

    for (i, block) in blocks.iter().enumerate() {
        let addr = block as *const _ as usize;
        assert_eq!(
            addr % 64, 0,
            "Block {} at address 0x{:x} should be 64-byte aligned", i, addr
        );
        assert_eq!(
            addr - base_addr, i * 64,
            "Block {} should be at offset {} from base", i, i * 64
        );
    }
}
