use criterion::Criterion;
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_io::stream::video::{ImageFormat, VideoCodec, VideoWriter};
use reqwest::blocking::get;
use std::{fs::File, hint::black_box, io::copy, path::PathBuf, time::Duration};
use tempfile::{tempdir, TempDir};

const FILE_NAME: &str = "video.mp4";
const VIDEO_LINK: &str =
    "https://github.com/kornia/tutorials/raw/refs/heads/master/data/sharpening.mp4";

fn download_video() -> (PathBuf, TempDir) {
    let response = get(VIDEO_LINK).expect("Failed to download video");
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let temp_file_path = temp_dir.path().join(FILE_NAME);
    let mut temp_file = File::create(&temp_file_path).expect("Failed to create temp file");

    copy(
        &mut response.bytes().expect("Failed to read response").as_ref(),
        &mut temp_file,
    )
    .expect("Failed to write video to temp file");

    println!("Video downloaded to: {temp_file_path:?}");
    (temp_file_path, temp_dir)
}

fn benchmark_get_buffer(c: &mut Criterion) {
    let (video_path, _temp_dir) = download_video();

    let pipeline_desc = format!(
        "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB,width=1024,height=688,framerate=8/1 ! appsink name=sink sync=true",
        video_path.to_str().unwrap()
    );

    let mut stream_capture = kornia_io::stream::StreamCapture::new(&pipeline_desc).unwrap();
    stream_capture.start().unwrap();

    c.bench_function("grab", |b| {
        b.iter(|| {
            black_box(stream_capture.grab_rgb8()).expect("Failed to grab the image");
        });
    });
}

fn benchmark_video_writer_push_1080p(c: &mut Criterion) {
    let tmp_dir = tempdir().expect("Failed to create temp directory");
    let file_path = tmp_dir.path().join("out_1080p.mp4");

    let size = ImageSize {
        width: 1920,
        height: 1080,
    };

    let mut writer =
        VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)
            .expect("Failed to create VideoWriter");
    writer.start().expect("Failed to start VideoWriter");

    let img = Image::<u8, 3, CpuAllocator>::new(
        size,
        vec![128u8; 1920 * 1080 * 3],
        CpuAllocator,
    )
    .expect("Failed to create image");

    c.bench_function("video_writer_push_1080p_rgb", |b| {
        b.iter(|| {
            black_box(writer.write(&img)).unwrap();
        });
    });

    writer.close().expect("Failed to close VideoWriter");
    drop(tmp_dir);
}

fn benchmark_video_writer_push_240p(c: &mut Criterion) {
    let tmp_dir = tempdir().expect("Failed to create temp directory");
    let file_path = tmp_dir.path().join("out_240p.mp4");

    let size = ImageSize {
        width: 320,
        height: 240,
    };

    let mut writer =
        VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)
            .expect("Failed to create VideoWriter");
    writer.start().expect("Failed to start VideoWriter");

    let img = Image::<u8, 3, CpuAllocator>::new(
        size,
        vec![128u8; 320 * 240 * 3],
        CpuAllocator,
    )
    .expect("Failed to create image");

    c.bench_function("video_writer_push_240p_rgb", |b| {
        b.iter(|| {
            black_box(writer.write(&img)).unwrap();
        });
    });

    writer.close().expect("Failed to close VideoWriter");
    drop(tmp_dir);
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(1));
    targets = benchmark_get_buffer, benchmark_video_writer_push_1080p, benchmark_video_writer_push_240p
}

criterion::criterion_main!(benches);
