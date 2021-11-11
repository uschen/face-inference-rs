// extern crate bindgen;
extern crate cmake;

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=libretinaface/src/**/*");
    println!("cargo:rerun-if-changed=libretinaface/CMakeLists.txt");

    // println!("cargo:rerun-if-changed=wrapper.h");

    let tensorrt_root = match env::var("TENSORRT_ROOT") {
        Ok(v) => v,
        Err(_) => String::from("/usr/local/lib"),
    };
    let cuda_root = match env::var("CUDA_HOME") {
        Ok(v) => v,
        Err(_) => String::from("/usr/local/cuda-10.2/targets/aarch64-linux/lib"),
    };

    const CURRENT_DIR: &str = "libretinaface";

    let cpp_libs = cmake::Config::new(CURRENT_DIR)
        .always_configure(true)
        .cxxflag("-DCMAKE_CUDA_FLAGS=\"--expt-extended-lambda -std=c++14\"")
        .build();

    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // println!("cargo:rustc-link-lib=dylib=stdc++");
    // println!("cargo:rustc-flags=-l dylib=stdc++");
    println!("cargo:rustc-link-search=native={}", cpp_libs.display());
    println!("cargo:rustc-link-search=native={}", &tensorrt_root);
    println!("cargo:rustc-link-search=native={}", &cuda_root);
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=static=retinaface");
}
