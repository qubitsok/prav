use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use xshell::{cmd, Shell};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Build and benchmark automation for prav-core", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Runs the performance_test example on the specified target
    Bench {
        #[arg(long, short, value_enum, default_value_t = Target::X86_64)]
        target: Target,

        /// Run with --release
        #[arg(long, default_value_t = true)]
        release: bool,

        /// Pin the benchmark process to a specific CPU core (using taskset)
        #[arg(long, short = 'p')]
        pin_core: Option<usize>,

        /// Select topology to benchmark
        #[arg(long, short = 'o', value_enum, default_value_t = TopologyArg::Square)]
        topology: TopologyArg,
    },
    /// Checks compilation for all supported targets
    CheckAll,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Target {
    /// Native x86_64
    X86_64,
    /// Linux on ARM64 (using cross)
    Aarch64,
    /// Bare-metal ARMv7R (Simulated Cortex-R5F on SK-KV260-G)
    Armv7r,
    /// WebAssembly
    Wasm32,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum TopologyArg {
    Square,
    Rectangle,
    Triangular,
    Honeycomb,
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match cli.command {
        Commands::Bench {
            target,
            release,
            pin_core,
            topology,
        } => {
            run_bench(&sh, target, release, pin_core, topology)?;
        }
        Commands::CheckAll => {
            check_all(&sh)?;
        }
    }

    Ok(())
}

fn run_bench(
    sh: &Shell,
    target: Target,
    release: bool,
    pin_core: Option<usize>,
    topology: TopologyArg,
) -> Result<()> {
    // Assume running from project root
    let profile_flag = if release { "--release" } else { "" };

    // Determine target directory location (workspace vs package root)
    let target_base = if std::path::Path::new("target").exists() {
        "target"
    } else {
        "../target"
    };

    // Helper to wrap command with taskset if requested
    let taskset_prefix = |core: Option<usize>| -> String {
        match core {
            Some(c) => format!("taskset -c {} ", c),
            None => String::new(),
        }
    };

    let prefix = taskset_prefix(pin_core);
    
    // Convert topology enum to string arg
    let topo_str = topology.to_possible_value().unwrap().get_name().to_string();
    let app_args = format!("-- --topology {}", topo_str);

    match target {
        Target::X86_64 => {
            println!(">> Benchmarking x86-64 (Native)...");
            if let Some(c) = pin_core {
                println!("   (Pinned to Core {})", c);
            }
            // Native run with specific CPU target flags
            let rust_flags = "-C target-cpu=x86-64";
            let _env = sh.push_env("RUSTFLAGS", rust_flags);

            println!("   [Clean] Cleaning bench-suite...");
            cmd!(sh, "cargo clean -p bench-suite").run()?;

            let cmd_str = format!("{}cargo run -p bench-suite {} {}", prefix, profile_flag, app_args);
            cmd!(sh, "bash -c {cmd_str}").run()?;
        }
        Target::Aarch64 => {
            println!(">> Benchmarking Aarch64 (via Cross)...");
            // Requires 'cross' installed
            ensure_cross(sh)?;

            // Clean before building to avoid GLIBC mismatches with cross container
            println!("   [Clean] Cleaning bench-suite...");
            cmd!(sh, "cargo clean").run()?;

            // Note: Pinning 'cross' client might not strictly pin the Docker container
            // unless the container runtime inherits affinity or we pass engine args.
            // This is a best-effort approach for cross.
            let cmd_str = format!(
                "{}cross run --target aarch64-unknown-linux-gnu -p bench-suite {} {}",
                prefix, profile_flag, app_args
            );
            cmd!(sh, "bash -c {cmd_str}").run()?;
        }
        Target::Armv7r => {
            println!(">> Benchmarking Armv7r (Simulated Cortex-R5F on SK-KV260-G)...");

            println!("   [Clean] Cleaning bench-suite...");
            cmd!(sh, "cargo clean -p bench-suite").run()?;

            // 1. Build for Native Cortex-R5F (using R5 + VFPv3-D16)
            println!("   [Step 1] Building for native Cortex-R5F (armv7r-none-eabi)...");
            let r5_target = "armv7r-none-eabi";
            ensure_target(sh, r5_target)?;
            // Rust doesn't recognize "cortex-r5f", so we use "cortex-r5" with FPU enabled manually.
            let r5_flags = "-C target-cpu=cortex-r5 -C target-feature=+vfp3d16";
            let _env_r5 = sh.push_env("RUSTFLAGS", r5_flags);
            cmd!(
                sh,
                "cargo build --target {r5_target} -p bench-suite {profile_flag}"
            )
            .run()?;
            drop(_env_r5);

            let release_dir = if release { "release" } else { "debug" };
            let binary_r5 = format!("{}/{}/{}/bench-suite", target_base, r5_target, release_dir);
            println!("   >> Native R5F build successful: {}", binary_r5);

            if cmd!(sh, "qemu-system-arm --version").read().is_err() {
                println!("!! 'qemu-system-arm' not found. Skipping execution.");
            } else {
                println!("--- Execution Output (QEMU: VersatilePB Cortex-R5F) ---");
                println!("   Binary: {}", binary_r5);

                // Execute QEMU manually to capture output and handle exit code 1 gracefully
                // We use -cpu cortex-r5f here because QEMU supports it directly.
                let qemu_cmd = format!(
                    "{}timeout 60s qemu-system-arm \
                    -M versatilepb -cpu cortex-r5f \
                    -nographic -display none \
                    -kernel {} \
                    -serial mon:stdio \
                    -semihosting",
                    prefix, binary_r5
                );
                
                // Note: We cannot easily pass CLI args to bare-metal QEMU via semihosting 
                // without complex setup (semihosting-args).
                // For now, we warn that topology selection is not supported on bare metal.
                if topology != TopologyArg::Square {
                    println!("!! Warning: Topology selection is not fully supported on bare-metal QEMU yet. Defaulting to compiled-in default.");
                }

                let output = cmd!(sh, "bash -c {qemu_cmd}").ignore_status().read()?;
                println!("{}", output);

                if output.contains("Solved:") {
                    // Success! QEMU might exit with 1 on some platforms via semihosting,
                    // but we consider "Solved:" as proof of completion.
                } else {
                    // Check if it failed with exit code
                    let status = cmd!(sh, "bash -c {qemu_cmd}").quiet().run();
                    if status.is_err() {
                        return Err(anyhow::anyhow!(
                            "QEMU failed or timed out without solution."
                        ));
                    }
                }
            }
        }
        Target::Wasm32 => {
            println!(">> Benchmarking Wasm32 (via Node.js)...");
            ensure_target(sh, "wasm32-unknown-unknown")?;
            ensure_wasm_bindgen(sh)?;

            println!("   [Clean] Cleaning bench-suite...");
            cmd!(sh, "cargo clean -p bench-suite").run()?;

            // 1. Build
            cmd!(
                sh,
                "cargo build --target wasm32-unknown-unknown -p bench-suite {profile_flag}"
            )
            .run()?;

            // 2. Bindgen
            let release_path = if release { "release" } else { "debug" };
            let wasm_path = format!(
                "{}/wasm32-unknown-unknown/{}/bench-suite.wasm",
                target_base, release_path
            );
            let out_dir = format!("{}/wasm-out", target_base);
            cmd!(
                sh,
                "wasm-bindgen {wasm_path} --out-dir {out_dir} --target nodejs --no-typescript"
            )
            .run()?;

            // 3. Run
            println!("--- Execution Output ---");
            if let Some(c) = pin_core {
                println!("   (Pinned Node to Core {})", c);
            }
            // Pass args to node script. Note: Node script might need to pass them to wasm?
            // wasm-bindgen for nodejs usually takes args?
            // Actually, `wasm-bindgen` generated JS doesn't automatically parse process.argv and pass to main.
            // We'd need to modify the JS wrapper or the Rust code to read usage of `std::env::args` in wasm.
            // But `std::env::args` in wasm32-unknown-unknown is usually empty or stubbed unless using WASI.
            // We are using `wasm-bindgen` with `nodejs` target.
            
            println!("!! Warning: Topology selection on Wasm32 (Node.js) requires extra setup. Ignoring arg.");
            
            let cmd_str = format!("{}node {}/bench-suite.js", prefix, out_dir);
            cmd!(sh, "bash -c {cmd_str}").run()?;
        }
    }
    Ok(())
}

fn check_all(sh: &Shell) -> Result<()> {
    // Assume running from project root
    ensure_cross(sh)?;

    println!("--- Checking x86-64 ---");
    let _env = sh.push_env("RUSTFLAGS", "-C target-cpu=x86-64");
    cmd!(sh, "cargo check --examples --release").run()?;
    drop(_env);

    println!("--- Checking Aarch64 ---");
    cmd!(
        sh,
        "cross check --target aarch64-unknown-linux-gnu --examples --release"
    )
    .run()?;

    println!("--- Checking Armv7r ---");
    ensure_target(sh, "armv7r-none-eabi")?;
    {
        // Check with FPU enabled to match bench config
        let _env_r5 = sh.push_env(
            "RUSTFLAGS",
            "-C target-cpu=cortex-r5 -C target-feature=+vfp3d16",
        );
        cmd!(
            sh,
            "cargo check --target armv7r-none-eabi -p bench-suite --release"
        )
        .run()?;
    }

    println!("--- Checking Wasm32 ---");
    ensure_target(sh, "wasm32-unknown-unknown")?;
    cmd!(
        sh,
        "cargo check --target wasm32-unknown-unknown -p bench-suite --release"
    )
    .run()?;

    println!(">> All targets checked successfully.");
    Ok(())
}

fn ensure_cross(sh: &Shell) -> Result<()> {
    if cmd!(sh, "cross --version").read().is_err() {
        println!("!! 'cross' is not installed. Installing via cargo...");
        cmd!(sh, "cargo install cross").run()?;
    }
    Ok(())
}

fn ensure_target(sh: &Shell, target: &str) -> Result<()> {
    let output = cmd!(sh, "rustup target list --installed").read()?;
    if !output.contains(target) {
        println!("!! Target '{}' not found. Installing via rustup...", target);
        cmd!(sh, "rustup target add {target}").run()?;
    }
    Ok(())
}

fn ensure_wasm_bindgen(sh: &Shell) -> Result<()> {
    if cmd!(sh, "wasm-bindgen --version").read().is_err() {
        println!("!! 'wasm-bindgen-cli' is not installed. Installing via cargo...");
        cmd!(sh, "cargo install wasm-bindgen-cli").run()?;
    }
    Ok(())
}
