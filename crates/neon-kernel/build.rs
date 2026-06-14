/// Build script for neon-kernel.
///
/// Compiles the Zig kernel (zig-kernel) and links it as a static library.
/// Falls back gracefully if Zig is not installed.

fn main() {
    // Paths
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = std::path::Path::new(&manifest_dir);
    let zig_kernel_dir = workspace_root.join("..").join("zig-kernel");
    let zig_build_dir = zig_kernel_dir.join("zig-out");

    // Only attempt Zig build if zig is available and we're on aarch64
    #[cfg(target_arch = "aarch64")]
    {
        // Try to run zig build
        let status = std::process::Command::new("zig")
            .args(&["build", "-Doptimize=ReleaseFast"])
            .current_dir(&zig_kernel_dir)
            .status();

        match status {
            Ok(status) if status.success() => {
                // Tell rustc where to find the static library
                let lib_path = zig_build_dir.join("lib");
                println!("cargo:rustc-link-search=native={}", lib_path.display());
                println!("cargo:rustc-link-lib=static=rotation_zig");
                println!("cargo:rerun-if-changed={}", zig_kernel_dir.join("src").join("main.zig").display());

                // Also copy the .a to the output directory for tests
                let out_dir = std::env::var("OUT_DIR").unwrap();
                let out_path = std::path::Path::new(&out_dir).join("librotation_zig.a");
                let src_a = lib_path.join("librotation_zig.a");
                if src_a.exists() {
                    std::fs::copy(&src_a, &out_path).unwrap_or_default();
                }
            }
            _ => {
                // Zig not available — fall back to pure Rust kernels.
                // This is fine: the Rust auto-vectorized versions work,
                // they're just ~3x slower on the NEON hot paths.
                println!("cargo:warning=zig not found; falling back to pure Rust kernels");
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("cargo:warning=non-ARM target; Zig NEON kernels skipped");
    }
}
