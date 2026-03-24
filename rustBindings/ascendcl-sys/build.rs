/// Build script for ascendcl-sys.
///
/// Locates the CANN SDK installation and links against libascendcl.so.
/// Set ASCEND_HOME env var to override the default search path.
///
/// With the "stub" feature, linking is skipped entirely.
fn main() {
    // Don't link if building in stub mode (CI, dev without CANN)
    if cfg!(feature = "stub") {
        println!("cargo:warning=ascendcl-sys: building in STUB mode, no libascendcl.so linked");
        return;
    }

    // Locate CANN SDK
    let ascend_home = std::env::var("ASCEND_HOME").unwrap_or_else(|_| {
        // Try common install paths
        let candidates = [
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/latest",
            "/opt/ascend/ascend-toolkit/latest",
        ];
        for path in &candidates {
            if std::path::Path::new(path).join("lib64/libascendcl.so").exists() {
                println!("cargo:warning=ascendcl-sys: auto-detected CANN at {}", path);
                return path.to_string();
            }
        }
        panic!(
            "CANN SDK not found. Set ASCEND_HOME env var or install to /usr/local/Ascend/ascend-toolkit/latest\n\
             Or build with --features stub to skip linking."
        );
    });

    let lib_dir = format!("{}/lib64", ascend_home);

    // Verify the library exists
    let lib_path = format!("{}/libascendcl.so", lib_dir);
    if !std::path::Path::new(&lib_path).exists() {
        panic!(
            "libascendcl.so not found at {}. Is CANN SDK installed correctly?\n\
             ASCEND_HOME = {}",
            lib_path, ascend_home
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=ascendcl");

    // Tell cargo to re-run if these change
    println!("cargo:rerun-if-env-changed=ASCEND_HOME");
    println!("cargo:rerun-if-changed=build.rs");
}
