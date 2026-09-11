// Temporary: modules are built incrementally and not all public items are wired
// into `main` yet. This is removed once the full pipeline is assembled.
#![allow(dead_code)]

mod features;
mod video;

fn main() {
    // Placeholder while modules are built incrementally.
    let _ = video::read_frames(std::path::Path::new("sample.mp4"), 1);
    let _extractor = features::make_extractor(features::DetectorKind::Orb, 2000);
}
