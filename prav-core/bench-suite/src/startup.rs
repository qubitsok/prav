

#[cfg(all(target_os = "none", target_abi = "eabi"))]
use core::arch::global_asm;

// Use panic_halt when we are on bare metal (no OS).
// This provides a default panic handler that just loops forever.
#[cfg(all(target_os = "none", target_abi = "eabi"))]
use panic_halt as _;

// --- ARMv7-R Startup Assembly ---
// Since we are running on bare metal without a standard runtime (crt0),
// we must define the entry point and the exception vector table ourselves.
// This is critical for the processor to know where to start executing code
// and how to handle interrupts/exceptions.

#[cfg(all(target_os = "none", target_abi = "eabi"))]
global_asm!(
    r#"
    .section .vector_table, "ax"
    .global _vectors
    _vectors:
        ldr pc, =_start                    // 0x00 Reset: Initial entry point
        ldr pc, =_asm_undefined_handler    // 0x04 Undefined Instruction
        ldr pc, =_asm_svc_handler          // 0x08 Supervisor Call (SVC)
        ldr pc, =_asm_prefetch_abort_handler // 0x0C Prefetch Abort (Instruction fetch error)
        ldr pc, =_asm_data_abort_handler   // 0x10 Data Abort (Data access error)
        b .                                // 0x14 Reserved
        ldr pc, =_asm_irq_handler          // 0x18 IRQ (Interrupt Request)
        ldr pc, =_asm_fiq_handler          // 0x1C FIQ (Fast Interrupt Request)

    .section .text.startup, "ax"
    .global _start
    .global _asm_undefined_handler
    .global _asm_svc_handler
    .global _asm_prefetch_abort_handler
    .global _asm_data_abort_handler
    .global _asm_irq_handler
    .global _asm_fiq_handler

    // UART FIFO address for VersatilePB (used for debug output in handlers)
    .equ UART0_FIFO, 0x101f1000
    .equ UART1_FIFO, 0x101f1000

    _start:
        // 1. Set Vector Base Address Register (VBAR) to our table.
        // This tells the CPU where to find the handlers defined above.
        ldr r0, =_vectors
        mcr p15, 0, r0, c12, c0, 0
        isb

        // 2. Initialize Stack Pointer.
        // We set the stack to a hardcoded address suitable for the board's memory map.
        // In a production environment, this symbol would come from the linker script.
        // ldr sp, =_stack_top
        ldr sp, =0x04200000 
        
        // 3. Jump to the Rust entry point.
        b Reset

    // --- Exception Handlers ---
    // These handlers just print a character to UART indicating the fault type and hang.
    // U = Undefined, S = SVC, P = Prefetch, D = Data, I = IRQ, F = FIQ

    _asm_undefined_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #85 // 'U'
        str r1, [r0]
        str r1, [r2]
        b .
    _asm_svc_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #83 // 'S'
        str r1, [r0]
        str r1, [r2]
        b .
    _asm_prefetch_abort_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #80 // 'P'
        str r1, [r0]
        str r1, [r2]
        b .
    _asm_data_abort_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #68 // 'D'
        str r1, [r0]
        str r1, [r2]
        b .
    _asm_irq_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #73 // 'I'
        str r1, [r0]
        str r1, [r2]
        b .
    _asm_fiq_handler:
        ldr r0, =UART0_FIFO
        ldr r2, =UART1_FIFO
        mov r1, #70 // 'F'
        str r1, [r0]
        str r1, [r2]
        b .
    "#
);
