use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // Put memory.x in the linker search path
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_default();

    println!("cargo:warning=Target is: {}", target);

    let memory_file = if target.contains("armv7r-none-eabi") {
        println!("cargo:warning=Using memory_r5.x");
        include_bytes!("memory_r5.x").as_slice()
    } else {
        println!("cargo:warning=Using memory.x");
        include_bytes!("memory.x").as_slice()
    };

    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(memory_file)
        .unwrap();

    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=memory_r5.x");
}
