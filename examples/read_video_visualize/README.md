# Kornia-rs: Video Reader Visualization Example

This example demonstrates the usage of `kornia_io::stream::video::VideoWriter` and `kornia_io::stream::video::VideoReader` to create a video, read it back, visualize frames and metadata using Rerun, extract specific frames, and test seeking.

## Prerequisites
*   Rust toolchain
*   GStreamer development libraries and essential plugins (e.g., `libav`, `x264enc`).
*   `rerun-cli` viewer (`pip install rerun-cli` or `cargo binstall rerun-cli`).

## Running the Example

Execute the following command from the root directory of the `kornia-rs` workspace:

```bash
cargo run --bin read_video_visualize