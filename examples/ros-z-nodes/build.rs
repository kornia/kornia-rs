use std::path::PathBuf;

/// Find all proto files in the protos directory
fn find_proto_files(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect("Failed to read protos directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? == "proto" {
                Some(path.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let proto_files = find_proto_files("protos");

    let mut prost_build = prost_build::Config::new();
    prost_build.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]");
    prost_build.file_descriptor_set_path(out_dir.join("descriptor.bin"));
    prost_build.compile_protos(&proto_files, &["protos/"])?;

    for proto in &proto_files {
        println!("cargo:rerun-if-changed={proto}");
    }
    println!("cargo:rerun-if-changed=protos");

    Ok(())
}
