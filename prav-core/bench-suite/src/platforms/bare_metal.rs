use super::{BenchmarkHost, Measurement};

// --- Hardware Constants ---

// SK-KV260-G RPU Frequency: 533 MHz
const FREQ_R5F: f64 = 533_000_000.0;

// --- Global Allocator ---
// We need a heap allocator because the QEC algorithm uses dynamic structures (Arena, Vec).
// `embedded_alloc` provides a simple linked-list allocator suitable for small embedded systems.
#[global_allocator]
static HEAP: embedded_alloc::Heap = embedded_alloc::Heap::empty();

/// Initializes the heap. Must be called at startup.
pub unsafe fn init_heap() {
    use core::mem::MaybeUninit;
    use core::ptr::addr_of_mut;

    // Allocate 256 KB for the heap in the BSS section.
    const HEAP_SIZE: usize = 1024 * 256;
    static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

    HEAP.init(addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE)
}

// --- Critical Section ---
// Required by many embedded crates to ensure thread safety (even if single-core).
#[cfg(any(target_abi = "eabihf", target_abi = "eabi"))]
struct MyCriticalSection;

#[cfg(any(target_abi = "eabihf", target_abi = "eabi"))]
critical_section::set_impl!(MyCriticalSection);

#[cfg(any(target_abi = "eabihf", target_abi = "eabi"))]
unsafe impl critical_section::Impl for MyCriticalSection {
    unsafe fn acquire() -> critical_section::RawRestoreState {}
    unsafe fn release(_token: critical_section::RawRestoreState) {}
}

// --- Panic Handler ---
#[cfg(target_abi = "eabihf")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

// --- 1. Cortex-M Implementation (Simulation) ---
// Used when running in QEMU simulating an M7 core (easier tooling).
#[cfg(all(target_arch = "arm", target_abi = "eabihf"))]
pub struct Platform;

#[cfg(all(target_arch = "arm", target_abi = "eabihf"))]
impl BenchmarkHost for Platform {
    type TimePoint = u32;

    fn now() -> Self::TimePoint {
        // Use SysTick current value
        cortex_m::peripheral::SYST::get_current()
    }

    fn measure(start: Self::TimePoint) -> Measurement {
        let end = Self::now();
        // SysTick counts DOWN.
        let ticks = if start >= end {
            start - end
        } else {
            // SysTick is 24-bit wrapped
            (0x00FFFFFF - end) + start
        };

        // We simulate the timing of the real R5F hardware based on cycle counts
        // to get an estimate of performance before deploying to the real board.
        let ticks_u64 = ticks as u64;
        let simulated_seconds = (ticks as f64) / FREQ_R5F;

        Measurement {
            ticks: Some(ticks_u64),
            micros: simulated_seconds * 1_000_000.0,
        }
    }

    fn print(s: &str) {
        use core::fmt::Write;
        use cortex_m_semihosting::hio;
        // Use Semihosting to print to the host console via the debugger/QEMU.
        if let Ok(mut stdout) = hio::hstdout() {
            let _ = stdout.write_str(s);
            let _ = stdout.write_str("\n");
        }
    }

    fn platform_name() -> &'static str {
        "SK-KV260-G RPU (Cortex-R5F @ 533MHz) [Simulated via M7]"
    }
}

// --- 2. Cortex-R Implementation (Native) ---
// Used when running on the actual ZynqMP RPU or QEMU emulating Cortex-R5.
#[cfg(all(target_arch = "arm", target_abi = "eabi"))]
pub struct Platform;

#[cfg(all(target_arch = "arm", target_abi = "eabi"))]
impl BenchmarkHost for Platform {
    type TimePoint = u32;

    fn now() -> Self::TimePoint {
        // Read the PMCCNTR (Performance Monitor Cycle Count Register).
        // This register gives us the exact number of CPU cycles executed.
        let cycles: u32;
        unsafe {
            // Memory barrier to ensure previous instructions are finished.
            core::arch::asm!("isb");
            core::arch::asm!(
                "mrc p15, 0, {}, c9, c13, 0",
                out(reg) cycles,
                options(nomem, nostack)
            );
        }
        cycles
    }

    fn measure(start: Self::TimePoint) -> Measurement {
        unsafe {
            core::arch::asm!("isb");
        }
        let end = Self::now();
        // PMCCNTR is a 32-bit counter that counts UP.
        let ticks = end.wrapping_sub(start);

        let ticks_u64 = ticks as u64;
        let elapsed_seconds = (ticks as f64) / FREQ_R5F;

        Measurement {
            ticks: Some(ticks_u64),
            micros: elapsed_seconds * 1_000_000.0,
        }
    }

    fn print(s: &str) {
        // Write directly to the UART FIFO register.
        // Address 0x101f1000 is the UART0 PL011 on the VersatilePB board (default QEMU ARM board).
        // Note: On the actual Xilinx board, this address would be different (e.g., Cadence UART).
        const UART0_FIFO: *mut u32 = 0x101f1000 as *mut u32;
        for b in s.bytes() {
            unsafe {
                core::ptr::write_volatile(UART0_FIFO, b as u32);
            }
        }
        unsafe {
            core::ptr::write_volatile(UART0_FIFO, b'\n' as u32);
        }
    }

    fn platform_name() -> &'static str {
        "Cortex-R5F (QEMU VersatilePB)"
    }
}
