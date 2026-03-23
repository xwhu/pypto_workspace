/// Build script for aclnn-sys.
///
/// Links against libopapi.so (aclnn operators) and libnnopbase.so.
fn main() {
    if cfg!(feature = "stub") {
        println!("cargo:warning=aclnn-sys: building in STUB mode, no libopapi.so linked");
        return;
    }

    let ascend_home = std::env::var("ASCEND_HOME").unwrap_or_else(|_| {
        let candidates = [
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/latest",
            "/opt/ascend/ascend-toolkit/latest",
        ];
        for path in &candidates {
            if std::path::Path::new(path).join("lib64/libopapi.so").exists() {
                return path.to_string();
            }
        }
        panic!(
            "CANN SDK not found (libopapi.so). Set ASCEND_HOME or build with --features stub."
        );
    });

    let lib_dir = format!("{}/lib64", ascend_home);

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=opapi");
    println!("cargo:rustc-link-lib=dylib=nnopbase");

    println!("cargo:rerun-if-env-changed=ASCEND_HOME");
    println!("cargo:rerun-if-changed=build.rs");
}
