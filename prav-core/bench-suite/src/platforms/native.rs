use super::{BenchmarkHost, Measurement};
use std::println;

/// Implementation for hosted environments (Linux, macOS, Windows).
/// Uses the standard library for high-precision timing and IO.
pub struct Platform;

impl BenchmarkHost for Platform {
    type TimePoint = std::time::Instant;

    fn now() -> Self::TimePoint {
        std::time::Instant::now()
    }

    fn measure(start: Self::TimePoint) -> Measurement {
        let elapsed = start.elapsed().as_secs_f64();
        Measurement {
            ticks: None, // Wall time is sufficient for OS-hosted benchmarks.
            micros: elapsed * 1_000_000.0,
        }
    }

    fn print(s: &str) {
        println!("{}", s);
    }

    fn platform_name() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        return "SK-KV260-G APU (Cortex-A53 @ 1.33GHz) [Linux]";
        #[cfg(target_arch = "x86_64")]
        return "Host (x86_64)";
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return "Host (Generic)";
    }
}
