/// Build script for hccl-sys.
///
/// Locates the CANN SDK installation and links against libhccl.so.
/// Set ASCEND_HOME env var to override the default search path.
///
/// With the "stub" feature, linking is skipped entirely.
fn main() {
    if cfg!(feature = "stub") {
        println!("cargo:warning=hccl-sys: building in STUB mode, no libhccl.so linked");
        return;
    }

    let ascend_home = std::env::var("ASCEND_HOME").unwrap_or_else(|_| {
        let candidates = [
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/latest",
            "/opt/ascend/ascend-toolkit/latest",
        ];
        for path in &candidates {
            if std::path::Path::new(path)
                .join("lib64/libhcomm.so")
                .exists()
            {
                println!("cargo:warning=hccl-sys: auto-detected CANN at {}", path);
                return path.to_string();
            }
        }
        panic!(
            "CANN SDK not found (looking for libhcomm.so). Set ASCEND_HOME env var or install to /usr/local/Ascend/ascend-toolkit/latest\n\
             Or build with --features stub to skip linking."
        );
    });

    let lib_dir = format!("{}/lib64", ascend_home);

    let lib_path = format!("{}/libhcomm.so", lib_dir);
    if !std::path::Path::new(&lib_path).exists() {
        panic!(
            "libhcomm.so not found at {}. Is CANN SDK installed correctly?\n\
             ASCEND_HOME = {}",
            lib_path, ascend_home
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=hccl");
    println!("cargo:rustc-link-lib=dylib=hcomm");

    println!("cargo:rerun-if-env-changed=ASCEND_HOME");
    println!("cargo:rerun-if-changed=build.rs");
}
