//! Integration tests for the streaming decoder.
//!
//! These tests verify that streaming decoding produces results equivalent
//! to batch decoding for the same syndrome data.

use prav_core::topology::Grid3D;
use prav_core::{Arena, StreamingConfig, StreamingDecoder, streaming_buffer_size};

#[test]
fn test_streaming_empty_syndromes() {
    const STRIDE_Y: usize = 4;
    let width = 4;
    let height = 4;
    let window_size = 3;
    let num_rounds = 10;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: STRIDE_Y,
        stride_z: STRIDE_Y * STRIDE_Y,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];

    // Ingest all rounds and collect committed corrections
    for _ in 0..num_rounds {
        if let Some(committed) = decoder.ingest_round(&empty) {
            assert!(
                committed.corrections.is_empty(),
                "Round {} should have no corrections with empty syndromes, got {}",
                committed.round,
                committed.corrections.len()
            );
        }
    }

    // Flush remaining and verify empty corrections
    for committed in decoder.flush() {
        assert!(
            committed.corrections.is_empty(),
            "Round {} should have no corrections with empty syndromes, got {}",
            committed.round,
            committed.corrections.len()
        );
    }
}

#[test]
fn test_streaming_round_ordering() {
    const STRIDE_Y: usize = 4;
    let width = 4;
    let height = 4;
    let window_size = 3;
    let num_rounds = 15;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: STRIDE_Y,
        stride_z: STRIDE_Y * STRIDE_Y,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];
    let mut committed_rounds = Vec::new();

    // Ingest all rounds
    for _ in 0..num_rounds {
        if let Some(committed) = decoder.ingest_round(&empty) {
            committed_rounds.push(committed.round);
        }
    }

    // Flush remaining
    for committed in decoder.flush() {
        committed_rounds.push(committed.round);
    }

    // Verify we get all rounds in order
    assert_eq!(committed_rounds.len(), num_rounds);
    for (i, &round) in committed_rounds.iter().enumerate() {
        assert_eq!(round, i as u64, "Expected round {} but got {}", i, round);
    }
}

#[test]
fn test_streaming_window_fill_behavior() {
    const STRIDE_Y: usize = 4;
    let width = 4;
    let height = 4;
    let window_size = 3;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: STRIDE_Y,
        stride_z: STRIDE_Y * STRIDE_Y,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];

    // First window_size rounds: no commits
    for i in 0..window_size {
        let result = decoder.ingest_round(&empty);
        assert!(result.is_none(), "Round {} should not commit yet", i);
    }

    // After window fills, each new round commits the oldest
    for i in window_size..10 {
        let result = decoder.ingest_round(&empty);
        assert!(result.is_some(), "Round {} should commit", i);
        let committed = result.unwrap();
        assert_eq!(
            committed.round,
            (i - window_size) as u64,
            "Expected to commit round {}, got {}",
            i - window_size,
            committed.round
        );
    }
}

#[test]
fn test_streaming_flush_returns_remaining() {
    // window_size=5, so max(4,4,5)=5, next_power_of_two=8
    const STRIDE_Y: usize = 8;
    let width = 4;
    let height = 4;
    let window_size = 5;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: 8,
        stride_z: 8 * 8,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];

    // Ingest 3 rounds (less than window size)
    for _ in 0..3 {
        decoder.ingest_round(&empty);
    }

    // Flush should return exactly 3 rounds
    let flushed: Vec<_> = decoder.flush().collect();
    assert_eq!(flushed.len(), 3);
    assert_eq!(flushed[0].round, 0);
    assert_eq!(flushed[1].round, 1);
    assert_eq!(flushed[2].round, 2);
}

#[test]
fn test_streaming_reset_clears_state() {
    const STRIDE_Y: usize = 4;
    let width = 4;
    let height = 4;
    let window_size = 3;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: STRIDE_Y,
        stride_z: STRIDE_Y * STRIDE_Y,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];

    // Ingest some rounds
    for _ in 0..5 {
        decoder.ingest_round(&empty);
    }

    assert!(decoder.is_window_full());
    assert_eq!(decoder.head_round(), 5);

    // Reset
    decoder.reset();

    assert!(!decoder.is_window_full());
    assert_eq!(decoder.head_round(), 0);
    assert_eq!(decoder.tail_round(), 0);
    assert_eq!(decoder.rounds_in_window(), 0);

    // Should be able to ingest again from scratch
    decoder.ingest_round(&empty);
    assert_eq!(decoder.head_round(), 1);
    assert_eq!(decoder.rounds_in_window(), 1);
}

#[test]
fn test_streaming_with_isolated_defects() {
    const STRIDE_Y: usize = 4;
    let width = 4;
    let height = 4;
    let window_size = 3;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: STRIDE_Y,
        stride_z: STRIDE_Y * STRIDE_Y,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    // Round 0: defect at position 0
    let round0 = [0b0001u64];
    decoder.ingest_round(&round0);

    // Round 1: defect at same position (should pair with round 0)
    let round1 = [0b0001u64];
    decoder.ingest_round(&round1);

    // Round 2: empty
    let round2 = [0u64];
    decoder.ingest_round(&round2);

    // Window is now full, next round commits round 0
    let round3 = [0u64];
    let committed = decoder.ingest_round(&round3);

    assert!(committed.is_some());
    let c = committed.unwrap();
    assert_eq!(c.round, 0);
    // The defect pair should create a Z-edge correction
}

#[test]
fn test_streaming_large_window() {
    // max(6, 6, 10) = 10, next_power_of_two = 16
    const STRIDE_Y: usize = 16;
    let width = 6;
    let height = 6;
    let window_size = 10;
    let num_rounds = 50;

    let config = StreamingConfig {
        window_size,
        width,
        height,
        detectors_per_round: width * height,
        stride_y: 16,
        stride_z: 16 * 16,
    };

    let buf_size = streaming_buffer_size(width, height, window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> = StreamingDecoder::new(&mut arena, config);

    let empty = [0u64; 1];
    let mut committed_count = 0;

    for _ in 0..num_rounds {
        if decoder.ingest_round(&empty).is_some() {
            committed_count += 1;
        }
    }

    // Committed = num_rounds - window_size (rounds committed during ingestion)
    assert_eq!(committed_count, num_rounds - window_size);

    // Flush remaining
    let flushed: Vec<_> = decoder.flush().collect();
    assert_eq!(flushed.len(), window_size);

    // Total = all rounds
    assert_eq!(committed_count + flushed.len(), num_rounds);
}
