// Temporary: modules are built incrementally and not all public items are wired
// into `main` yet. This is removed once the full pipeline is assembled.
#![allow(dead_code)]

mod features;
mod matching;
mod ply_writer;
mod reconstruction;
mod video;

#[cfg(test)]
mod test_util;

fn main() {
    // Placeholder while modules are built incrementally.
    let _ = video::read_frames(std::path::Path::new("sample.mp4"), 1);
    let _extractor = features::make_extractor(features::DetectorKind::Orb, 2000);
    let _edges: Vec<kornia_calib::TrackEdge> = matching::match_sequential_pairs(&[], 5, 0.8);
    let _cam = reconstruction::make_camera(600.0, 600.0, 320.0, 240.0);
    let _verts: Vec<ply_writer::Vertex> = Vec::new();
}
