mod video;

fn main() {
    // Placeholder while modules are built incrementally.
    let _ = video::read_frames(std::path::Path::new("sample.mp4"), 1);
}