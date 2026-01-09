//! # Statistics and Percentile Calculation Utilities
//!
//! This module provides statistical tools for analyzing benchmark results:
//!
//! - **ThresholdPoint**: Results for a single (distance, error_rate) configuration
//! - **SuppressionFactor**: Lambda (Λ) calculation for threshold analysis
//! - **LatencyStats**: Timing percentiles (p50, p95, p99)
//! - **Wilson CI**: Confidence intervals for binomial proportions
//!
//! ## Key Metrics
//!
//! ### Logical Error Rate (LER)
//!
//! The fraction of decoding attempts that resulted in a logical error:
//!
//! ```text
//! LER = logical_errors / (num_shots × num_rounds)
//! ```
//!
//! We normalize by rounds because more rounds = more chances for error.
//!
//! ### Error Suppression Factor (Lambda, Λ)
//!
//! How much better a larger code performs compared to a smaller one:
//!
//! ```text
//! Λ = LER(distance d) / LER(distance d+2)
//! ```
//!
//! - **Λ > 1**: Larger code has lower LER → below threshold ✓
//! - **Λ < 1**: Larger code has higher LER → above threshold ✗
//! - **Λ = 1**: At threshold
//!
//! ### Confidence Intervals
//!
//! We use the Wilson score interval for binomial proportions. It's better
//! than the normal approximation when the number of successes is small.
//!
//! For z = 1.96 (95% confidence):
//!
//! ```text
//! center = (p + z²/2n) / (1 + z²/n)
//! spread = z × sqrt((p(1-p) + z²/4n) / n) / (1 + z²/n)
//! CI = [center - spread, center + spread]
//! ```

use std::time::Duration;

/// Results from a single (distance, error_rate) benchmark configuration.
///
/// This is the primary output structure. It contains everything needed
/// to analyze decoder performance at one point in parameter space.
///
/// ## Key Fields
///
/// - `ler_per_round`: The main metric - logical error rate normalized by rounds
/// - `ler_ci_low`, `ler_ci_high`: 95% confidence interval for the LER
/// - `decode_time_us`: Average decoding latency in microseconds
///
/// ## Usage
///
/// ```ignore
/// let point = ThresholdPoint::new(5, 0.003, 10000, 5, 100, 0.15);
///
/// println!("d={} p={:.4}: LER={:.2e} [{:.2e},{:.2e}]",
///     point.distance,
///     point.physical_error_rate,
///     point.ler_per_round,
///     point.ler_ci_low,
///     point.ler_ci_high);
/// ```
#[derive(Debug, Clone)]
pub struct ThresholdPoint {
    /// Code distance (d). Determines grid size: (d-1) × (d-1) × d.
    pub distance: usize,

    /// Physical error rate (p). The per-gate or per-timestep error probability.
    pub physical_error_rate: f64,

    /// Number of syndrome samples processed.
    pub num_shots: usize,

    /// Number of measurement rounds per shot (usually equals distance).
    pub num_rounds: usize,

    /// Number of logical errors observed (predicted != actual).
    pub logical_errors: usize,

    /// Logical error rate per round: `errors / (shots × rounds)`.
    /// This normalization allows comparison across different round counts.
    pub ler_per_round: f64,

    /// Lower bound of 95% Wilson confidence interval for LER.
    pub ler_ci_low: f64,

    /// Upper bound of 95% Wilson confidence interval for LER.
    pub ler_ci_high: f64,

    /// Average decoding time in microseconds.
    pub decode_time_us: f64,
}

impl ThresholdPoint {
    /// Create a new threshold point from raw benchmark results.
    ///
    /// Automatically computes:
    /// - LER per round = logical_errors / (num_shots × num_rounds)
    /// - 95% Wilson confidence interval for the LER
    ///
    /// # Parameters
    ///
    /// - `distance`: Code distance
    /// - `physical_error_rate`: Error rate being tested
    /// - `num_shots`: Number of syndrome samples
    /// - `num_rounds`: Rounds per shot
    /// - `logical_errors`: Number of logical errors observed
    /// - `decode_time_us`: Average decode time in microseconds
    pub fn new(
        distance: usize,
        physical_error_rate: f64,
        num_shots: usize,
        num_rounds: usize,
        logical_errors: usize,
        decode_time_us: f64,
    ) -> Self {
        // Logical error rate per round
        let total_rounds = num_shots * num_rounds;
        let ler_per_round = if total_rounds > 0 {
            logical_errors as f64 / total_rounds as f64
        } else {
            0.0
        };

        // Wilson score 95% CI (z = 1.96)
        let (ci_low, ci_high) = wilson_ci(logical_errors, total_rounds, 1.96);

        Self {
            distance,
            physical_error_rate,
            num_shots,
            num_rounds,
            logical_errors,
            ler_per_round,
            ler_ci_low: ci_low,
            ler_ci_high: ci_high,
            decode_time_us,
        }
    }

    /// Format as CSV row.
    pub fn to_csv(&self) -> String {
        format!(
            "{},{:.6},{},{},{},{:.6e},{:.6e},{:.6e},{:.3}",
            self.distance,
            self.physical_error_rate,
            self.num_rounds,
            self.num_shots,
            self.logical_errors,
            self.ler_per_round,
            self.ler_ci_low,
            self.ler_ci_high,
            self.decode_time_us,
        )
    }
}

/// Error suppression factor Lambda (Λ) between two code distances.
///
/// Lambda measures how much error suppression we get from increasing
/// the code distance. It's the key metric for threshold analysis.
///
/// ## Interpretation
///
/// - **Λ > 1**: Error rate decreased → below threshold (good!)
/// - **Λ < 1**: Error rate increased → above threshold (bad!)
/// - **Λ = 1**: At threshold
///
/// ## Calculation
///
/// ```text
/// Λ = LER(d_low) / LER(d_high)
/// ```
///
/// For surface codes, Λ should be approximately constant below threshold,
/// with Λ ≈ exp((d_high - d_low) / ξ) where ξ is the correlation length.
///
/// ## Error Propagation
///
/// We propagate uncertainty using the standard formula for ratios:
///
/// ```text
/// δΛ/Λ = sqrt((δε_low/ε_low)² + (δε_high/ε_high)²)
/// ```
#[derive(Debug, Clone)]
pub struct SuppressionFactor {
    /// Lower code distance in the comparison.
    pub d_low: usize,

    /// Higher code distance in the comparison.
    pub d_high: usize,

    /// Physical error rate at which Lambda was computed.
    pub physical_error_rate: f64,

    /// Lambda value: LER(d_low) / LER(d_high).
    /// Values > 1 indicate error suppression.
    pub lambda: f64,

    /// Uncertainty in lambda from error propagation.
    pub lambda_err: f64,
}

impl SuppressionFactor {
    /// Compute Lambda from two threshold points at adjacent distances.
    ///
    /// Returns `None` if either LER is zero (can't divide by zero).
    ///
    /// # Parameters
    ///
    /// - `low`: Results from the smaller code distance
    /// - `high`: Results from the larger code distance
    ///
    /// # Returns
    ///
    /// `Some(SuppressionFactor)` with computed lambda and uncertainty,
    /// or `None` if computation is not possible.
    pub fn from_points(low: &ThresholdPoint, high: &ThresholdPoint) -> Option<Self> {
        if low.ler_per_round <= 0.0 || high.ler_per_round <= 0.0 {
            return None;
        }

        let lambda = low.ler_per_round / high.ler_per_round;

        // Error propagation: δΛ/Λ = sqrt((δε_low/ε_low)² + (δε_high/ε_high)²)
        let rel_err_low = (low.ler_ci_high - low.ler_ci_low) / (2.0 * low.ler_per_round);
        let rel_err_high = (high.ler_ci_high - high.ler_ci_low) / (2.0 * high.ler_per_round);
        let rel_err = (rel_err_low.powi(2) + rel_err_high.powi(2)).sqrt();
        let lambda_err = lambda * rel_err;

        Some(Self {
            d_low: low.distance,
            d_high: high.distance,
            physical_error_rate: low.physical_error_rate,
            lambda,
            lambda_err,
        })
    }
}

/// Wilson score confidence interval for binomial proportions.
///
/// This is a better alternative to the normal approximation, especially
/// when the number of successes is small or close to 0 or trials.
///
/// # Formula
///
/// ```text
/// p̂ = successes / trials
/// center = (p̂ + z²/2n) / (1 + z²/n)
/// spread = z × sqrt((p̂(1-p̂) + z²/4n) / n) / (1 + z²/n)
/// CI = [center - spread, center + spread]
/// ```
///
/// # Parameters
///
/// - `successes`: Number of positive outcomes (logical errors)
/// - `trials`: Total number of attempts (shots × rounds)
/// - `z`: Z-score for desired confidence level
///   - z = 1.96 → 95% confidence
///   - z = 2.576 → 99% confidence
///
/// # Returns
///
/// (lower, upper) bounds for the true probability. Clamped to [0, 1].
///
/// # Example
///
/// ```ignore
/// // 50 errors out of 10,000 trials, 95% CI
/// let (low, high) = wilson_ci(50, 10000, 1.96);
/// // low ≈ 0.0037, high ≈ 0.0065
/// ```
pub fn wilson_ci(successes: usize, trials: usize, z: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }

    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;

    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let spread = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt() / denom;

    ((center - spread).max(0.0), (center + spread).min(1.0))
}

/// CSV header for threshold study output.
///
/// Columns: distance, physical_p, rounds, shots, logical_errors,
/// ler_per_round, ler_ci_low, ler_ci_high, decode_us
pub const CSV_HEADER: &str = "distance,physical_p,rounds,shots,logical_errors,ler_per_round,ler_ci_low,ler_ci_high,decode_us";

/// Latency statistics for decoding performance analysis.
///
/// Contains common percentiles useful for understanding decoder performance:
/// - Average and extremes (avg, min, max)
/// - Percentiles for tail latency (p50, p95, p99)
///
/// All values are in microseconds (µs).
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Average (mean) decode time in microseconds.
    pub avg_us: f64,

    /// Minimum decode time observed.
    pub min_us: f64,

    /// Maximum decode time observed.
    pub max_us: f64,

    /// Median (50th percentile) decode time.
    pub p50_us: f64,

    /// 95th percentile decode time (tail latency).
    pub p95_us: f64,

    /// 99th percentile decode time (extreme tail).
    pub p99_us: f64,
}

/// Calculate latency statistics from a list of decode times.
///
/// Computes average, min, max, and percentiles (p50, p95, p99).
///
/// # Parameters
///
/// - `times`: Vector of decode durations
///
/// # Returns
///
/// `LatencyStats` with all times converted to microseconds.
/// Returns all zeros if `times` is empty.
pub fn calculate_percentiles(times: &[Duration]) -> LatencyStats {
    if times.is_empty() {
        return LatencyStats {
            avg_us: 0.0,
            min_us: 0.0,
            max_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
        };
    }

    let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();

    LatencyStats {
        avg_us: sum / n as f64,
        min_us: sorted[0],
        max_us: sorted[n - 1],
        p50_us: percentile(&sorted, 50.0),
        p95_us: percentile(&sorted, 95.0),
        p99_us: percentile(&sorted, 99.0),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Format a large number with K/M/G suffixes.
pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_percentiles() {
        let times: Vec<Duration> = (1..=100).map(Duration::from_micros).collect();

        let stats = calculate_percentiles(&times);
        assert!((stats.avg_us - 50.5).abs() < 0.1);
        assert_eq!(stats.min_us, 1.0);
        assert_eq!(stats.max_us, 100.0);
        // p50 should be around 50, allowing for rounding differences
        assert!((stats.p50_us - 50.0).abs() < 2.0, "p50={}", stats.p50_us);
        assert!((stats.p99_us - 99.0).abs() < 2.0, "p99={}", stats.p99_us);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_500_000), "1.5M");
        assert_eq!(format_number(1_500_000_000), "1.5G");
    }

    #[test]
    fn test_wilson_ci_basic() {
        // 50% success rate with 100 trials
        let (low, high) = wilson_ci(50, 100, 1.96);
        assert!(low > 0.39 && low < 0.41, "low={}", low);
        assert!(high > 0.59 && high < 0.61, "high={}", high);
    }

    #[test]
    fn test_wilson_ci_edge_cases() {
        // Zero trials
        let (low, high) = wilson_ci(0, 0, 1.96);
        assert_eq!(low, 0.0);
        assert_eq!(high, 1.0);

        // Zero successes
        let (low, high) = wilson_ci(0, 100, 1.96);
        assert_eq!(low, 0.0);
        assert!(high > 0.0 && high < 0.05, "high={}", high);

        // All successes
        let (low, high) = wilson_ci(100, 100, 1.96);
        assert!(low > 0.95 && low < 1.0, "low={}", low);
        assert!((high - 1.0).abs() < 1e-10, "high={}", high);
    }

    #[test]
    fn test_threshold_point() {
        let point = ThresholdPoint::new(5, 0.003, 10000, 5, 100, 0.15);

        assert_eq!(point.distance, 5);
        assert_eq!(point.num_shots, 10000);
        assert_eq!(point.num_rounds, 5);
        assert_eq!(point.logical_errors, 100);

        // LER per round = 100 / (10000 * 5) = 0.002
        assert!((point.ler_per_round - 0.002).abs() < 1e-6);

        // CI should bracket the point estimate
        assert!(point.ler_ci_low < point.ler_per_round);
        assert!(point.ler_ci_high > point.ler_per_round);
    }

    #[test]
    fn test_suppression_factor() {
        let low = ThresholdPoint::new(3, 0.005, 10000, 3, 150, 0.05);
        let high = ThresholdPoint::new(5, 0.005, 10000, 5, 50, 0.15);

        let lambda = SuppressionFactor::from_points(&low, &high).unwrap();

        assert_eq!(lambda.d_low, 3);
        assert_eq!(lambda.d_high, 5);

        // low LER = 150/(10000*3) = 0.005
        // high LER = 50/(10000*5) = 0.001
        // Lambda = 0.005/0.001 = 5.0
        assert!(
            (lambda.lambda - 5.0).abs() < 0.01,
            "lambda={}",
            lambda.lambda
        );
        assert!(lambda.lambda_err > 0.0);
    }
}
