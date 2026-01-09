//! Tests for the DecoderBuilder API.
//!
//! The builder pattern provides a type-safe way to construct decoders
//! without manually calculating the STRIDE_Y const generic.

use prav_core::arena::Arena;
use prav_core::decoder::builder::DecoderBuilder;
use prav_core::decoder::types::EdgeCorrection;
use prav_core::required_buffer_size;
use prav_core::topology::SquareGrid;

#[test]
fn test_builder_basic_construction() {
    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(16, 16)
        .build(&mut arena);

    assert!(
        decoder.is_ok(),
        "Builder should succeed for valid dimensions"
    );
}

#[test]
fn test_builder_no_dimensions_error() {
    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let result = DecoderBuilder::<SquareGrid>::new().build(&mut arena);

    assert!(
        result.is_err(),
        "Builder should fail when dimensions not set"
    );
}

#[test]
fn test_builder_3d_construction() {
    let size = required_buffer_size(8, 8, 8);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions_3d(8, 8, 8)
        .build(&mut arena);

    assert!(decoder.is_ok(), "3D builder should succeed");
}

#[test]
fn test_builder_stride_calculation() {
    // Test various dimension combinations and their expected strides
    let cases = [
        ((8, 8, 1), 8),
        ((16, 16, 1), 16),
        ((17, 17, 1), 32),
        ((32, 32, 1), 32),
        ((64, 64, 1), 64),
        ((8, 8, 8), 8),
        ((9, 9, 9), 16),
    ];

    for ((w, h, d), expected_stride) in cases {
        let builder = DecoderBuilder::<SquareGrid>::new().dimensions_3d(w, h, d);
        assert_eq!(
            builder.stride_y(),
            expected_stride,
            "Stride for ({},{},{}) should be {}",
            w,
            h,
            d,
            expected_stride
        );
    }
}

#[test]
fn test_builder_different_strides() {
    // Test builder can handle different stride requirements

    // Stride 8
    {
        let size = required_buffer_size(8, 8, 1);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);
        let decoder = DecoderBuilder::<SquareGrid>::new()
            .dimensions(8, 8)
            .build(&mut arena);
        assert!(decoder.is_ok());
    }

    // Stride 16
    {
        let size = required_buffer_size(16, 16, 1);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);
        let decoder = DecoderBuilder::<SquareGrid>::new()
            .dimensions(16, 16)
            .build(&mut arena);
        assert!(decoder.is_ok());
    }

    // Stride 32
    {
        let size = required_buffer_size(32, 32, 1);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);
        let decoder = DecoderBuilder::<SquareGrid>::new()
            .dimensions(32, 32)
            .build(&mut arena);
        assert!(decoder.is_ok());
    }
}

#[test]
fn test_dyn_decoder_load_syndromes() {
    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(16, 16)
        .build(&mut arena)
        .unwrap();

    // Create empty syndromes
    let num_blocks = (16usize * 16).div_ceil(64);
    let syndromes = vec![0u64; num_blocks];

    // Load syndromes through the dyn decoder
    decoder.load_dense_syndromes(&syndromes);

    // Should not panic
}

#[test]
fn test_dyn_decoder_decode() {
    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(16, 16)
        .build(&mut arena)
        .unwrap();

    // Create syndromes with two adjacent defects
    let num_blocks = (16usize * 16).div_ceil(64);
    let mut syndromes = vec![0u64; num_blocks];
    syndromes[0] = 0b11; // Two adjacent defects

    decoder.load_dense_syndromes(&syndromes);

    let mut corrections = vec![EdgeCorrection::default(); 100];
    let count = decoder.decode(&mut corrections);

    assert!(count >= 1, "Should produce corrections for defect pair");
}

#[test]
fn test_dyn_decoder_full_reset() {
    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(16, 16)
        .build(&mut arena)
        .unwrap();

    // Load some syndromes
    let syndromes = vec![0u64; 4];
    decoder.load_dense_syndromes(&syndromes);

    // Full reset should work
    decoder.full_reset();
}

#[test]
fn test_builder_non_square_grid() {
    // Rectangular grid where height > width
    let size = required_buffer_size(8, 32, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(8, 32)
        .build(&mut arena);

    assert!(decoder.is_ok(), "Should support rectangular grids");
}

#[test]
fn test_builder_with_observables() {
    use prav_core::decoder::observables::ObservableMode;

    let size = required_buffer_size(16, 16, 1);
    let mut buffer = vec![0u8; size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(16, 16)
        .build(&mut arena)
        .unwrap();

    // Set and get observable mode
    decoder.set_observable_mode(ObservableMode::Phenomenological);
    let mode = decoder.observable_mode();
    assert!(
        matches!(mode, ObservableMode::Phenomenological),
        "Observable mode should be Phenomenological"
    );

    // Get predicted observables
    let predicted = decoder.predicted_observables();
    assert_eq!(predicted, 0, "Initial predicted observables should be 0");
}
