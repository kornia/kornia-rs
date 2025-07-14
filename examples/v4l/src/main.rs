#[cfg(target_os = "linux")]
mod v4l;
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    v4l::v4l_demo()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("This example is only supported on Linux due to V4L dep.");
}
