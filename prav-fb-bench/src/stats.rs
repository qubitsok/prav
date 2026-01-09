//! Statistics calculations for latency percentiles.

use std::time::Duration;

/// Latency statistics with percentiles.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub avg_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

/// Calculate latency percentiles from a list of durations.
pub fn calculate_percentiles(times: &[Duration]) -> LatencyStats {
    if times.is_empty() {
        return LatencyStats {
            avg_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
        };
    }

    // Convert to microseconds
    let mut us: Vec<f64> = times
        .iter()
        .map(|d| d.as_secs_f64() * 1_000_000.0)
        .collect();

    // Sort for percentile calculation
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = us.len();
    let avg = us.iter().sum::<f64>() / n as f64;

    // Percentile indices
    let p50_idx = (n as f64 * 0.50) as usize;
    let p95_idx = (n as f64 * 0.95) as usize;
    let p99_idx = (n as f64 * 0.99) as usize;

    LatencyStats {
        avg_us: avg,
        p50_us: us[p50_idx.min(n - 1)],
        p95_us: us[p95_idx.min(n - 1)],
        p99_us: us[p99_idx.min(n - 1)],
    }
}

/// Format a number with thousand separators.
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(*c);
    }

    result
}
