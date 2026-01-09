/// A measurement result from a benchmark.
#[derive(Clone, Copy)]
pub struct Measurement {
    /// Raw hardware ticks (if available/relevant).
    #[allow(dead_code)] // Used by bare-metal platforms for cycle counting
    pub ticks: Option<u64>,
    /// Elapsed time in microseconds.
    pub micros: f64,
}

/// A trait that abstracts the host platform's capabilities.
///
/// This allows us to run the exact same QEC benchmark code on:
/// 1. A powerful x86_64 workstation (using `std::time`).
/// 2. A WebAssembly runtime in the browser (using `performance.now()`).
/// 3. A bare-metal ARM Cortex-R5 processor (using hardware cycle counters).
pub trait BenchmarkHost {
    /// The type representing a point in time.
    /// - `std::time::Instant` on Native.
    /// - `f64` (milliseconds) on Wasm.
    /// - `u32` (cycle count) on Bare Metal.
    type TimePoint: Copy;

    /// Returns the current time point.
    fn now() -> Self::TimePoint;

    /// Calculates the duration between `start` and now.
    fn measure(start: Self::TimePoint) -> Measurement;

    /// Prints a string to the platform's standard output.
    /// - `stdout` on Native/Wasm.
    /// - UART or Semihosting on Bare Metal.
    fn print(s: &str);

    /// Returns a human-readable name of the platform.
    fn platform_name() -> &'static str;
}

// Re-export specific implementations based on the target.

#[cfg(all(not(target_os = "none"), not(target_arch = "wasm32")))]
mod native;
#[cfg(all(not(target_os = "none"), not(target_arch = "wasm32")))]
pub use native::Platform;

#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::Platform;

#[cfg(target_os = "none")]
pub mod bare_metal;
#[cfg(target_os = "none")]
pub use bare_metal::Platform;
