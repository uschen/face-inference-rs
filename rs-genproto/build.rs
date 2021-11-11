use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use glob::glob;

const OUT_DIR: &str = "./src/prost";

fn main() {
    // clean up
    if fs::metadata(OUT_DIR).is_ok() {
        fs::remove_dir_all(OUT_DIR).expect("Failed to remove output dir");
    }
    // iterator all the potential folders
    let _protos: Vec<_> = glob("../proto/*.proto")
        .expect("failed to find .proto files")
        .filter_map(Result::ok)
        .map(|e| format!("{}", e.to_string_lossy()))
        .collect();

    for f in &_protos {
        println!("cargo:rerun-if-changed={}", f);
    }

    // create out dir.
    fs::create_dir_all(OUT_DIR).unwrap_or_else(|e| panic!("unable to create dir: {}", e));

    tonic_build::configure()
        .build_server(true)
        .out_dir(OUT_DIR)
        .compile(
            &_protos.iter().collect::<Vec<_>>(),
            &[&"../proto".to_string(), &"../thirdparty".to_string()],
        )
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {}", e));

    if let Err(_err) = fs::remove_file(format!("{}/google.protobuf.rs", OUT_DIR)) {}
    if let Err(_err) = fs::remove_file(format!("{}/google.api.rs", OUT_DIR)) {}

    let file_names: Vec<_> = fs::read_dir(OUT_DIR)
        .expect("read dir not ok")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .filter(|d| {
            if let Some(e) = d.path().extension() {
                e == "rs"
            } else {
                false
            }
        })
        .map(|e| format!("{}", e.file_name().to_string_lossy()))
        .collect();

    // rustfmt
    for rs_file in file_names {
        let mut file_name = PathBuf::new();
        file_name.push(OUT_DIR);
        file_name.push(rs_file);
        rustfmt(&file_name);
    }
}

fn rustfmt(file_path: &Path) {
    let output = Command::new("rustfmt")
        .arg(file_path.to_str().unwrap())
        .output();
    if !output.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("Rustfmt failed");
    }
}
