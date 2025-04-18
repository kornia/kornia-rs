use kornia_image::{Image, ImageSize};
use kornia_io::stream::video::{ImageFormat, VideoCodec, VideoReader, VideoWriter};
use log::{error, info, warn, trace};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::io::Write;
use std::env;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::builder().filter_level(log::LevelFilter::Info).init(); // Initialize logger

    // --- Create Temporary Video ---
    info!("Creating temporary video file...");
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let example_out_dir = Path::new(&target_dir).join("read_video_visualize_output");
    fs::create_dir_all(&example_out_dir)?;
    let file_path = example_out_dir.join("example_video.mp4");
    let output_dir = example_out_dir.join("extracted_frames");
    fs::create_dir_all(&output_dir)?;

    let size = ImageSize { width: 320, height: 240 };
    let fps = 30;
    let num_frames = 90; // 3 seconds

    // --- VideoWriter Block ---
    info!("Starting VideoWriter setup...");
    { // Scope for writer
        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::H264,
            ImageFormat::Rgb8,
            fps,
            size,
        )?;
        info!("VideoWriter created, starting pipeline...");
        writer.start()?;
        info!("VideoWriter pipeline started, writing frames...");

        for i in 0..num_frames {
            trace!("Preparing frame {}", i); // Use trace for high-frequency logs
            let mut data = Vec::with_capacity(size.width * size.height * 3);
            for y in 0..size.height {
                for x in 0..size.width {
                    let r = ((x as f32 / size.width as f32 * 255.0) + (i as f32 * 2.0)) as u8 % 255;
                    let g = ((y as f32 / size.height as f32 * 255.0) + (i as f32 * 1.5)) as u8 % 255;
                    let b = ((x as f32 + y as f32) / ((size.width + size.height) as f32) * 255.0 + (i as f32 * 1.0)) as u8 % 255;
                    data.push(r); data.push(g); data.push(b);
                }
            }
            let img = Image::<u8, 3>::new(size, data)?;

            // --- Match block for writer.write ---
            match writer.write(&img) {
                Ok(_) => {
                    // Frame written successfully, continue loop.
                    trace!("Wrote frame {}", i); // Progress log
                }
                Err(write_err) => { // Correctly use write_err variable
                    error!("Error during VideoWriter::write() on frame {}: {:?}", i, write_err);
                    return Err(Box::new(write_err)); // Return the boxed error
                }
            } // --- End Corrected Match block ---
        } // End of for loop

        info!("Finished writing frames. Attempting to close VideoWriter...");
        writer.close()?; // Handle potential close error
        info!("VideoWriter closed successfully.");
        info!("Temporary video created at: {:?}", file_path);

        info!("Waiting briefly for filesystem after close...");
        std::thread::sleep(std::time::Duration::from_millis(500)); // Wait 0.5 seconds

        // Check if file exists and has size after closing
        if !file_path.exists() || fs::metadata(&file_path)?.len() == 0 {
            error!("Video file was not created or is empty after closing writer!");
            return Err("VideoWriter failed to produce a valid output file.".into());
        } else {
            info!("Verified video file exists and is not empty.");
        }

    } // Writer scope ends

    // --- Initialize Rerun ---
    info!("Initializing Rerun...");
    let rec = rerun::RecordingStreamBuilder::new("kornia_rs/video_reader")
        .spawn()?;
    info!("Rerun viewer spawned.");


    // --- VideoReader Block ---
    info!("Opening video with VideoReader: {:?}", file_path);
    let mut reader = VideoReader::new(&file_path)?;
    info!("VideoReader created, starting pipeline...");
    reader.start()?;
    info!("VideoReader pipeline started.");


    // --- Log Metadata ---
    info!("Logging video properties to Rerun...");
    rec.log_static("video_info", &rerun::TextLog::new(format!( // Changed entity path slightly
                                                               "Video Properties:\n  Path: {:?}\n  Size: {}x{}\n  FPS: {:.2}\n  Format: {:?}\n  Duration: {:.2}s",
                                                               file_path,
                                                               reader.size().width, reader.size().height,
                                                               reader.fps(),
                                                               reader.format(),
                                                               reader.duration().unwrap_or(0.0)
    )).with_level(rerun::TextLogLevel::INFO))?;


    // --- Read Frames Loop ---
    let mut frame_count = 0;
    let mut extracted_count = 0;
    let extract_interval_secs = 0.5;
    let mut next_extract_time = 0.0;
    info!("Reading frames and logging to Rerun viewer...");

    loop { // Use loop instead of while let to handle Ok(None) explicitly
        match reader.read::<3>() {
            Ok(Some(img)) => { // Got a frame
                frame_count += 1;
                trace!("Read frame {}", frame_count);

                rec.set_time_sequence("frame", frame_count);
                let timestamp_secs = (frame_count - 1) as f64 / reader.fps(); // -1 because frame_count is 1-based
                if reader.fps() > 0.0 { // Avoid division by zero if FPS detection failed
                    rec.set_time_seconds("timestamp", timestamp_secs);
                }


                let tensor = rerun::TensorData::new(
                    vec![
                        img.size().height as u64,
                        img.size().width as u64,
                        3,
                    ],
                    rerun::datatypes::TensorBuffer::U8(img.as_slice().into()),
                );

                trace!("Attempting to log frame {} to Rerun...", frame_count);
                rec.log("video/frame", &rerun::Tensor::new(tensor))?;
                trace!("Successfully logged frame {}.", frame_count);


                // Frame Extraction Logic
                if reader.fps() > 0.0 && timestamp_secs >= next_extract_time && extracted_count < 5 {
                    let frame_path = output_dir.join(format!("frame_{:03}.ppm", extracted_count));
                    if let Err(e) = save_ppm(&frame_path, &img) {
                        error!("Failed to save frame {}: {}", extracted_count, e);
                    } else {
                        info!("Saved frame {} at {:.2}s to {:?}", extracted_count, timestamp_secs, frame_path);
                        rec.log("video/extracted_frame_marker", &rerun::Points2D::new([(0.0, 0.0)]).with_radii([0.0]))?;
                    }
                    extracted_count += 1;
                    next_extract_time += extract_interval_secs;
                }
            }
            Ok(None) => { // End of stream OR timeout without EOS flag yet
                if reader.is_eos() {
                    info!("End of stream reached according to reader.");
                    break; // Exit loop on EOS
                } else {
                    // Timeout without EOS
                    trace!("Reader returned None, but EOS flag not set. Looping again.");
                    // Small sleep to prevent tight loop on stall
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            Err(e) => { // Error during read
                error!("Error during VideoReader::read(): {:?}", e);
                // stop processing on a read error
                return Err(Box::new(e));
            }
        } // End match reader.read()
    } // End loop

    info!("Finished reading video ({} frames).", frame_count);
    info!("Extracted {} frames to {:?}", extracted_count, output_dir);


    // --- Seek Test ---
    if let Some(duration) = reader.duration() {
        if duration > 1.0 {
            info!("Testing seek functionality...");
            if let Err(e) = reader.seek(1.0) {
                error!("Seek operation failed: {:?}", e);
                // Don't necessarily exit, maybe just skip seek test log
            } else {
                info!("Seek successful, attempting to read frame...");
                match reader.read::<3>() { // Explicitly handle result after seek
                    Ok(Some(img)) => {
                        info!("Logging frame after seeking to 1.0s");
                        rec.set_time_sequence("frame", frame_count + 1); // Use a distinct frame number
                        rec.set_time_seconds("timestamp", 1.0);

                        let tensor = rerun::TensorData::new(
                            vec![
                                img.size().height as u64,
                                img.size().width as u64,
                                3,
                            ],
                            rerun::datatypes::TensorBuffer::U8(img.as_slice().into()),
                        );
                        rec.log("video/frame_after_seek", &rerun::Tensor::new(tensor))?;
                    }
                    Ok(None) => { warn!("Could not read frame after seeking (EOS or timeout)."); }
                    Err(e) => { error!("Error reading frame after seeking: {:?}", e); }
                }
            }
        }
    }


    // --- Clean up ---
    info!("Closing VideoReader...");
    reader.close()?; // Handle potential close error
    info!("VideoReader closed. Rerun viewer will remain open.");
    info!("You can close the Rerun window manually.");

    Ok(())
}

// --- save_ppm function ---
fn save_ppm(path: impl AsRef<Path>, image: &Image<u8, 3>) -> Result<(), Box<dyn Error>> {
    let width = image.size().width;
    let height = image.size().height;
    let data = image.as_slice();
    let mut file = std::fs::File::create(path)?;
    writeln!(&mut file, "P6")?;
    writeln!(&mut file, "{} {}", width, height)?;
    writeln!(&mut file, "255")?;
    file.write_all(data)?;
    Ok(())
}