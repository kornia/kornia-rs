use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Build CXX bridge
    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++14")
        .compile("kornia-cpp");

    println!("cargo:rerun-if-changed=src/lib.rs");

    // Generate version header from Cargo.toml
    generate_version_header();
}

fn generate_version_header() {
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    let version_hpp = format!(
        r#"#pragma once

// Auto-generated from Cargo.toml version
// Do not edit manually!

#define KORNIA_VERSION "{version}"
#define KORNIA_VERSION_MAJOR {major}
#define KORNIA_VERSION_MINOR {minor}
#define KORNIA_VERSION_PATCH {patch}

namespace kornia {{
namespace detail {{

inline const char* get_version() {{
    return KORNIA_VERSION;
}}

}} // namespace detail
}} // namespace kornia
"#,
        version = version,
        major = version.split('.').nth(0).unwrap_or("0"),
        minor = version.split('.').nth(1).unwrap_or("0"),
        patch = version.split('.').nth(2).unwrap_or("0"),
    );

    // Write to OUT_DIR for Rust build
    fs::write(out_dir.join("version.hpp"), &version_hpp)
        .expect("Failed to write version.hpp");

    // Also write to include directory for development
    let include_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("include")
        .join("kornia");
    
    if let Err(e) = fs::create_dir_all(&include_dir) {
        eprintln!("Warning: Could not create include dir: {}", e);
    }
    
    if let Err(e) = fs::write(include_dir.join("version.hpp"), &version_hpp) {
        eprintln!("Warning: Could not write version.hpp to include dir: {}", e);
    }
    
    println!("cargo:rerun-if-changed=Cargo.toml");
}

