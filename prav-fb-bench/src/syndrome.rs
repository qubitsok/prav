//! Syndrome generation and format conversion.
//!
//! Generates random syndromes in prav format (bitpacked u64) and
//! converts to fusion-blossom format (sparse vertex list).

use fusion_blossom::util::VertexIndex;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Generate random syndromes in prav format (bitpacked u64 arrays).
///
/// Uses power-of-2 stride for memory alignment matching prav-core.
pub fn generate_prav_syndromes(
    width: usize,
    height: usize,
    error_prob: f64,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<u64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Calculate stride (power of 2)
    let max_dim = width.max(height);
    let stride = max_dim.next_power_of_two();

    // Calculate number of u64 blocks needed
    let alloc_size = height * stride;
    let num_blocks = alloc_size.div_ceil(64);

    let mut syndromes = Vec::with_capacity(num_shots);

    for _ in 0..num_shots {
        let mut packed = vec![0u64; num_blocks];

        for y in 0..height {
            for x in 0..width {
                if rng.r#gen::<f64>() < error_prob {
                    let idx = y * stride + x;
                    let block = idx / 64;
                    let bit = idx % 64;
                    packed[block] |= 1u64 << bit;
                }
            }
        }

        syndromes.push(packed);
    }

    syndromes
}

/// Convert prav-format syndromes to fusion-blossom sparse format.
///
/// Returns a list of defect vertex indices in row-major order.
pub fn prav_to_fusion_blossom(packed: &[u64], width: usize, height: usize) -> Vec<VertexIndex> {
    let stride = width.max(height).next_power_of_two();
    let mut defects = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x;
            let block = idx / 64;
            let bit = idx % 64;

            if block < packed.len() && (packed[block] & (1u64 << bit)) != 0 {
                // Row-major index for fusion-blossom
                defects.push((y * width + x) as VertexIndex);
            }
        }
    }

    defects
}

/// Count number of defects in prav-format syndrome array.
pub fn count_defects(packed: &[u64]) -> usize {
    packed.iter().map(|x| x.count_ones() as usize).sum()
}
