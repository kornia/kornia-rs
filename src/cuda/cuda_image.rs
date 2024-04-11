use crate::image::{Image, ImageSize};

use anyhow::Result;
use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const PTX_SRC: &str = "
extern \"C\" __global__ void rgb_to_gray(const float* rgb, float* gray, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = rgb[3 * i + 0];
        float g = rgb[3 * i + 1];
        float b = rgb[3 * i + 2];
        float gray_val = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[3 * i + 0] = gray_val;
        gray[3 * i + 1] = gray_val;
        gray[3 * i + 2] = gray_val;
    }
}
";

pub struct CudaImage {
    pub data: CudaSlice<f32>,
    pub size: ImageSize,
    pub dev: Arc<CudaDevice>,
}

impl CudaImage {
    pub fn from_host(img: Image<f32, 3>, dev: Arc<CudaDevice>) -> Result<Self> {
        let img_host = img.data.as_slice().expect("Image data is not contiguous");
        let img_dev = dev.htod_sync_copy(&img_host)?;
        Ok(Self {
            data: img_dev,
            size: img.size(),
            dev,
        })
    }

    pub fn to_host(&self) -> Result<Image<f32, 3>> {
        let img_host = self.dev.dtoh_sync_copy(&self.data)?;
        Ok(Image::new(self.size, img_host)?)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        let data = self.dev.htod_sync_copy(&vec![0f32; self.data.len()])?;
        Ok(Self {
            data,
            size: self.size,
            dev: self.dev.clone(),
        })
    }
}

pub struct KorniaScript {
    pub dev: Arc<CudaDevice>,
    pub f: Arc<LaunchAsync>,
}

impl KorniaScript {
    pub fn new(dev: Arc<CudaDevice>, ptx_src: &str, kernel_name: &str) -> Result<Self> {
        let ptx = compile_ptx(ptx_src)?;

        dev.load_ptx(ptx, "kernel", &[kernel_name])?;

        let f = dev
            .get_func("kernel", kernel_name)
            .expect("Kernel not found");

        Ok(Self { dev, f })
    }

    pub fn run(&self, &img_cuda: &CudaImage, &mut gray_cuda: &mut CudaImage) -> Result<()> {
        let n = img_cuda.data.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (n / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.f
                .launch(cfg, (&img_cuda.data, &mut gray_cuda.data, n))?
        };
        Ok(())
    }
}

pub struct GramFromRgb {}

pub fn gray_from_rgb(image: &CudaImage) -> Result<CudaImage> {
    let ptx = compile_ptx(PTX_SRC)?;

    image.dev.load_ptx(ptx, "kernel", &["rgb_to_gray"])?;
    let f = image
        .dev
        .get_func("kernel", "rgb_to_gray")
        .expect("Kernel not found");

    let mut output = CudaImage {
        data: image.dev.htod_sync_copy(&vec![0f32; image.data.len()])?,
        size: image.size,
        dev: image.dev.clone(),
    };

    let n = output.data.len() as u32;
    //let cfg = LaunchConfig::for_num_elems(n);
    let cfg = LaunchConfig {
        grid_dim: (n / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&image.data, &mut output.data, n))? };

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_image() -> Result<()> {
        let img = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: 4,
                height: 5,
            },
            1f32,
        )?;

        let dev = CudaDevice::new(0)?;
        let img_cuda = CudaImage::from_host(img, dev)?;

        Ok(())
    }
}
