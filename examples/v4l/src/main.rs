#[cfg(target_os = "linux")]
mod webcam;
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    webcam::v4l_demo()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("This example is only supported on Linux due to V4L dep.");
}
