use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection, Peeling};
use prav_core::topology::SquareGrid;
use rand::prelude::*;
use rand::rngs::StdRng;

#[test]
fn test_edge_compaction_fuzz() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let width = 32;
    let height = 32;
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, width, height, 1);

    let mut rng = StdRng::seed_from_u64(12345);
    let mut corrections = vec![EdgeCorrection::default(); 10000];

    // Fuzz test: Emit random edges multiple times
    // Track expected state in a simplistic way (HashMap or sorted vec)
    let mut expected_edges = std::collections::HashMap::new();

    for _ in 0..1000 {
        let u = rng.random_range(0..width * height) as u32;
        // Pick a neighbor
        let dir = rng.random_range(0..4);
        let v = match dir {
            0 => {
                if (u % width as u32) > 0 {
                    u - 1
                } else {
                    continue;
                }
            }
            1 => {
                if (u % width as u32) < (width as u32 - 1) {
                    u + 1
                } else {
                    continue;
                }
            }
            2 => {
                if u >= width as u32 {
                    u - width as u32
                } else {
                    continue;
                }
            }
            3 => {
                if u < ((height - 1) * width) as u32 {
                    u + width as u32
                } else {
                    continue;
                }
            }
            _ => continue,
        };

        // Emit edge (u, v)
        decoder.emit_linear(u, v);

        // Update expected
        let (min, max) = if u < v { (u, v) } else { (v, u) };
        let count = expected_edges.entry((min, max)).or_insert(0);
        *count += 1;
    }

    // Also test boundary edges
    for _ in 0..100 {
        let u = rng.random_range(0..width * height) as u32;
        decoder.emit_linear(u, u32::MAX);
        let count = expected_edges.entry((u, u32::MAX)).or_insert(0);
        *count += 1;
    }

    // Reconstruct
    let count = decoder.reconstruct_corrections(&mut corrections);
    let result = &corrections[0..count];

    // Verify
    let mut reconstructed_set = std::collections::HashSet::new();
    for edge in result {
        reconstructed_set.insert((edge.u, edge.v));
    }

    for ((u, v), c) in expected_edges {
        if c % 2 == 1 {
            assert!(
                reconstructed_set.contains(&(u, v)),
                "Missing edge ({}, {})",
                u,
                v
            );
        } else {
            assert!(
                !reconstructed_set.contains(&(u, v)),
                "Extra edge ({}, {})",
                u,
                v
            );
        }
    }
}
