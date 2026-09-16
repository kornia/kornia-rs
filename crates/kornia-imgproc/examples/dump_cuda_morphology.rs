//! Dump kornia CPU and CUDA morphology results for OpenCV comparison.
//!
//! ```text
//! cargo run -p kornia-imgproc --example dump_cuda_morphology --features cuda --release -- dilate 64 48
//! ```

use kornia_image::{Image, ImageSize};
use kornia_imgproc::morphology::{dilate, erode, Kernel, KernelShape};
use kornia_imgproc::padding::PaddingMode;

fn usage() -> ! {
    eprintln!("Usage: dump_cuda_morphology dilate|erode WIDTH HEIGHT");
    std::process::exit(1);
}

fn print_array(values: &[u8]) {
    print!("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            print!(",");
        }
        print!("{value}");
    }
    print!("]");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        usage();
    }
    let operation = args[0].as_str();
    if !matches!(operation, "dilate" | "erode") {
        usage();
    }
    let width: usize = args[1].parse().unwrap_or_else(|_| usage());
    let height: usize = args[2].parse().unwrap_or_else(|_| usage());
    let size = ImageSize { width, height };
    let source_data: Vec<u8> = (0..width * height).map(|i| (i * 37 % 251) as u8).collect();
    let source = Image::<u8, 1>::new(size, source_data).expect("source image");
    let kernel = Kernel::new(KernelShape::Box { size: 3 });

    let mut cpu = Image::<u8, 1>::from_size_val(size, 0).expect("CPU destination");
    match operation {
        "dilate" => {
            dilate(&source, &mut cpu, &kernel, PaddingMode::Replicate, [0]).expect("CPU dilation")
        }
        "erode" => {
            erode(&source, &mut cpu, &kernel, PaddingMode::Replicate, [0]).expect("CPU erosion")
        }
        _ => unreachable!(),
    }

    let context = std::sync::Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = context.default_stream();
    let device_source = source.to_cuda(&stream).expect("H2D source");
    let mut device_dst = Image::<u8, 1>::zeros_cuda(size, &stream).expect("device destination");
    match operation {
        "dilate" => dilate(
            &device_source,
            &mut device_dst,
            &kernel,
            PaddingMode::Replicate,
            [0],
        )
        .expect("CUDA dilation"),
        "erode" => erode(
            &device_source,
            &mut device_dst,
            &kernel,
            PaddingMode::Replicate,
            [0],
        )
        .expect("CUDA erosion"),
        _ => unreachable!(),
    }
    let gpu = device_dst.to_host_image(&stream).expect("D2H destination");

    print!(r#"{{"operation":"{operation}","width":{width},"height":{height},"source":"#);
    print_array(source.as_slice());
    print!(r#", "cpu":"#);
    print_array(cpu.as_slice());
    print!(r#", "gpu":"#);
    print_array(gpu.as_slice());
    println!("}}");
}
