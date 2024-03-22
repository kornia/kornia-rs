use anyhow::Result;
use gst::prelude::*;
use kornia_rs::image::Image;
use std::sync::atomic::{AtomicUsize, Ordering};

struct WebcamCapture {
    pipeline: gst::Pipeline,
    _appsink: gst_app::AppSink,
    receiver: tokio::sync::mpsc::Receiver<(usize, Image<u8, 3>)>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl WebcamCapture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        gst::init()?;

        // TODO: parameterize the pipeline string
        let pipeline_str = r#"v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! videoscale ! video/x-raw,format=RGB ! appsink name=sink emit-signals=true"#;
        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("Failed to downcast pipeline"))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow::anyhow!("Failed to get sink"))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSink"))?;

        let (tx, rx) = tokio::sync::mpsc::channel(10);

        // initialize frame counter
        let frame_counter = AtomicUsize::new(0);

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    if let Ok(sample) = sink.pull_sample() {
                        let caps = sample.caps().expect("Failed to get caps from sample");
                        let video_info = gst_video::VideoInfo::from_caps(&caps)
                            .expect("Failed to get video info from caps");

                        // Extract width and height
                        let width = video_info.width() as usize;
                        let height = video_info.height() as usize;

                        if let Some(buffer) = sample.buffer() {
                            let map = buffer.map_readable().unwrap();
                            //let timestamp = buffer.pts().unwrap().nseconds();
                            // Create a new image from the buffer
                            let img = Image::<_, 3>::new(
                                kornia_rs::image::ImageSize { width, height },
                                map.as_slice().to_vec(),
                            );
                            match img {
                                Ok(img) => {
                                    // TODO: improve this
                                    let frame_number = frame_counter.fetch_add(1, Ordering::SeqCst);
                                    if let Err(_) = tx.blocking_send((frame_number, img)) {
                                        println!("Failed to send");
                                    }
                                }
                                Err(e) => {
                                    println!("Failed to create image: {:?}", e);
                                }
                            }
                        }
                    }
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        pipeline.set_state(gst::State::Playing)?;

        // Event loop or processing here
        // Set pipeline state to Null when done

        let bus = pipeline
            .bus()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bus"))?;

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Eos(..) => break,
                    MessageView::Error(err) => {
                        eprintln!(
                            "Error from {:?}: {}",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            //err.error_code()
                        );
                        break;
                    }
                    _ => (),
                }
            }
        });
        Ok(Self {
            pipeline,
            _appsink: appsink,
            receiver: rx,
            handle: Some(handle),
        })
    }

    async fn run<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut((usize, Image<u8, 3>)),
    {
        let receiver = &mut self.receiver;
        while let Some(data) = receiver.recv().await {
            match data {
                _ => f(data),
            }
        }

        Ok(())
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        println!("Dropping WebcamCapture");
        self.pipeline.set_state(gst::State::Null).unwrap();
        self.handle.take().map(|t| t.join());
        println!("Dropped WebcamCapture");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut webcam = WebcamCapture::new()?;
    let rec = rerun::RecordingStreamBuilder::new("Kornia App")
        .batcher_config(rerun::log::DataTableBatcherConfig {
            flush_num_bytes: 10,
            ..Default::default()
        })
        .spawn()?;
    //.connect()?;

    webcam
        .run(|(frame_number, img)| {
            //println!(
            //    "Received image #{} with size: {:?}",
            //    frame_number,
            //    img.image_size()
            //);

            let img = img.cast_and_scale::<f32>(1. / 255.).unwrap();

            let gray = kornia_rs::color::gray_from_rgb(&img).unwrap();
            let gray = kornia_rs::threshold::threshold_binary(&gray, 0.5, 1.0).unwrap();
            let _ = rec.log("gray", &rerun::Image::try_from(gray.data).unwrap());
            //let _ = rec.log_timeless("image", &rerun::Image::try_from(gray.data).unwrap());
        })
        .await?;

    Ok(())
}
