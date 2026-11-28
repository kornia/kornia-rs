#![cfg(feature = "gstreamer")]

use kornia_io::gstreamer::StreamCapture;
use std::time::{Duration, Instant};

#[test]
fn slow_consumer_must_receive_all_frames() -> Result<(), Box<dyn std::error::Error>> {
    let mut capture = StreamCapture::new("videotestsrc num-buffers=20 ! video/x-raw,format=RGB,width=8,height=4,framerate=30/1 ! appsink name=sink sync=false max-buffers=2 drop=false")?;
    capture.start()?;
    std::thread::sleep(Duration::from_millis(100));
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut count = 0;
    loop {
        if capture.grab_rgb8()?.is_some() {
            count += 1;
            std::thread::sleep(Duration::from_millis(2));
        } else if capture.is_eos() {
            while capture.grab_rgb8()?.is_some() {
                count += 1;
            }
            break;
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(Instant::now() < deadline);
    }
    capture.close()?;
    assert_eq!(count, 20, "slow consumer lost decoded frames");
    Ok(())
}
