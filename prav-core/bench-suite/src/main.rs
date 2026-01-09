#![no_std]
#![cfg_attr(target_os = "none", no_main)]

//! # Quantum Error Correction Benchmark Suite
//!
//! This crate contains performance benchmarks for the `prav-core` QEC decoder.
//! It supports multiple targets:
//! - **Native (x86_64, AArch64)**: Uses standard OS facilities for timing and output.
//! - **WebAssembly (Wasm32)**: Runs in browsers or Node.js, using the Performance API.
//! - **Bare Metal (ARM Cortex-R5, Cortex-M7)**: Runs directly on hardware/simulators
//!   without an OS, using hardware cycle counters and UART/Semihosting for output.

extern crate alloc;

#[cfg(not(target_os = "none"))]
extern crate std;

// --- Modules ---
mod benchmark;
mod platforms;
#[cfg(target_os = "none")]
mod startup; // Assembly startup code for bare metal

#[cfg(not(target_os = "none"))]
use alloc::borrow::ToOwned;
#[cfg(not(target_os = "none"))]
use alloc::format;
#[cfg(not(target_os = "none"))]
use std::string::String;

use crate::benchmark::{run_suite, TopologyArg};
use crate::platforms::{BenchmarkHost, Platform};

#[cfg(not(target_os = "none"))]
use clap::Parser;

// --- CLI Arguments (Host Only) ---
#[cfg(not(target_os = "none"))]
#[derive(Parser)]
struct Cli {
    /// Topology to benchmark (Square, Rectangle, Triangular, Honeycomb, All)
    #[arg(short, long, value_enum, default_value_t = TopologyArg::Square)]
    topology: TopologyArg,

    /// Run all grid sizes (including very large ones)
    #[arg(long, default_value_t = false)]
    all_grids: bool,
}

#[cfg(target_os = "none")]
const CYCLES_INFO: usize = 100;
#[cfg(not(target_os = "none"))]
const CYCLES_INFO: usize = 10_000;

// --- Main Entry Point: Host (OS) ---
#[cfg(not(target_os = "none"))]
fn main() {
    let args = Cli::parse();
    Platform::print("[Performance Test] 32x32 Grid (Stride 32)");
    Platform::print(Platform::platform_name());

    Platform::print(&format!("Cycles: {}", CYCLES_INFO));
    Platform::print(
        "----------------------------------------------------------------------------------",
    );
    run_suite(args.topology, args.all_grids);
}

// --- Main Entry Point: Cortex-M (Simulation) ---
#[cfg(all(target_os = "none", target_abi = "eabihf"))]
use cortex_m_rt::entry;

#[cfg(all(target_os = "none", target_abi = "eabihf"))]
#[entry]
fn main() -> ! {
    // 1. Enable SysTick for timing (used by Platform::now())
    {
        use cortex_m::peripheral::syst::SystClkSource;
        let mut peripherals = unsafe { cortex_m::Peripherals::steal() };
        peripherals.SYST.set_clock_source(SystClkSource::Core);
        peripherals.SYST.set_reload(0x00FFFFFF);
        peripherals.SYST.clear_current();
        peripherals.SYST.enable_counter();
    }

    // 2. Initialize Heap
    unsafe {
        platforms::bare_metal::init_heap();
    }

    Platform::print("[Performance Test] 32x32 Grid (Stride 32)");
    Platform::print(Platform::platform_name());

    // Topology selection via CLI is not supported on bare metal; default to Square.
    run_suite(TopologyArg::Square, false);

    Platform::print("Bare Metal Done");

    loop {}
}

// --- Main Entry Point: Cortex-R (Native) ---
// The actual entry point `_start` is defined in `startup.rs`.
// It jumps to `Reset`, which then calls `main`.

#[cfg(all(target_os = "none", target_abi = "eabi"))]
#[no_mangle]
pub unsafe extern "C" fn Reset() -> ! {
    main()
}

#[cfg(all(target_os = "none", target_abi = "eabi"))]
fn main() -> ! {
    // 1. Initialize Hardware (PMU, FPU)
    unsafe {
        // Enable PMU Cycle Counter (PMCCNTR)
        // PMCR: Enable all counters (Bit 0), Reset cycle counter (Bit 2)
        core::arch::asm!("mcr p15, 0, {}, c9, c12, 0", in(reg) 5u32);

        // PMCNTENSET: Enable Cycle Counter (Bit 31)
        core::arch::asm!("mcr p15, 0, {}, c9, c12, 1", in(reg) 0x80000000u32);

        // PMUSERENR: Enable User Mode Access to PMU (Bit 0)
        core::arch::asm!("mcr p15, 0, {}, c9, c14, 0", in(reg) 1u32);

        // Barriers to ensure PMU is running
        core::arch::asm!("dsb", "isb");

        // Enable FPU (CP10/CP11) in CPACR
        // Set bits 20-23 (CP10 and CP11) to 0b11 (Full Access)
        let mut cpacr: u32;
        core::arch::asm!("mrc p15, 0, {}, c1, c0, 2", out(reg) cpacr);
        cpacr |= 0xF00000;
        core::arch::asm!("mcr p15, 0, {}, c1, c0, 2", in(reg) cpacr);
        core::arch::asm!("isb");

        // Enable FPU in FPEXC (Bit 30)
        let mut fpexc: u32;
        core::arch::asm!("vmrs {}, fpexc", out(reg) fpexc);
        fpexc |= 0x40000000;
        core::arch::asm!("vmsr fpexc, {}", in(reg) fpexc);
        core::arch::asm!("isb");
    }

    // 2. Initialize Heap
    unsafe {
        platforms::bare_metal::init_heap();
    }

    Platform::print("Bare Metal Start");
    Platform::print(Platform::platform_name());

    run_suite(TopologyArg::Square, false);

    // 3. Exit via Semihosting (SYS_EXIT)
    // ADP_Stopped_ApplicationExit = 0x20026
    let block: [u32; 2] = [0x20026, 0];
    unsafe {
        core::arch::asm!(
            "mov r0, #0x18",
            "mov r1, {}",
            "svc 0x123456",
            in(reg) &block,
            options(nostack, noreturn)
        );
    }
}
