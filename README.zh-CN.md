# kornia-rs：基于 Rust 的高性能计算机视觉库

[English](README.md) | 简体中文

![Crates.io Version](https://img.shields.io/crates/v/kornia)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![Documentation](https://img.shields.io/badge/docs.rs-kornia-orange)](https://docs.rs/kornia)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)

`kornia` crate 是一个用 [Rust](https://www.rust-lang.org/) 🦀 编写的底层计算机视觉库。

使用该库可以在你的机器学习和数据科学项目中，以线程安全和高效的方式进行图像 I/O、可视化及其他底层操作。

## 快速开始

`cargo run --bin hello_world -- --image-path path/to/image.jpg`

```rust
use kornia::image::Image;
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 读取图片
    let image: Image<u8, 3, _> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
```

```bash
Hello, world! 🦀
Loaded Image size: ImageSize { width: 258, height: 195 }

Goodbyte!
```

## 功能特性

- 🦀库主要用 [Rust](https://www.rust-lang.org/) 编写。
- 🚀 多线程高效的图像 I/O、图像处理和高级计算机视觉算子。
- 🔢 高效的张量和图像 API，适用于深度学习和科学计算。
- 🐍 Python 绑定通过 [PyO3/Maturin](https://github.com/PyO3/maturin) 创建。
- 📦 支持 Linux [amd64/arm64]、Macos 和 Windows。
- 支持的 Python 版本有 3.7/3.8/3.9/3.10/3.11/3.12/3.13，包括 free-threaded 构建。

### 支持的图像格式

- 支持读取 AVIF、BMP、DDS、Farbeld、GIF、HDR、ICO、JPEG (libjpeg-turbo)、OpenEXR、PNG、PNM、TGA、TIFF、WebP 格式的图片。

### 图像处理

- 支持图像灰度化、缩放、裁剪、旋转、翻转、填充、归一化、反归一化等多种图像处理操作。

### 视频处理

- 支持摄像头视频帧捕获和视频写入。

## 🛠️ 安装

### >_ 系统依赖

根据你需要的功能，可能需要在系统中安装以下依赖：

#### v4l（Video4Linux 摄像头支持）

```bash
sudo apt-get install clang
```

#### turbojpeg

```bash
sudo apt-get install nasm
```

#### gstreamer

```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

** 查看 gstreamer 安装指南：<https://docs.rs/gstreamer/latest/gstreamer/#installation>

### 🦀 Rust

在你的 `Cargo.toml` 中添加如下内容：

```toml
[dependencies]
kornia = "0.1"
```

或者，你也可以单独使用各个子 crate：

```toml
[dependencies]
kornia-tensor = "0.1"
kornia-tensor-ops = "0.1"
kornia-io = "0.1"
kornia-image = "0.1"
kornia-imgproc = "0.1"
kornia-3d = "0.1"
kornia-apriltag = "0.1"
kornia-vlm = "0.1"
kornia-nn = "0.1"
kornia-algebra = "0.1"
```

### 🐍 Python

```bash
pip install kornia-rs
```

仅暴露了 Rust API 的子集。详见 [kornia 文档](https://kornia.readthedocs.io/en/stable/)，了解 `kornia-rs` Python 模块中暴露的函数和对象。

`kornia-rs` 库在 free-threaded Python 构建下是线程安全的。

## 示例：图像处理

以下示例展示如何读取一张图片，将其转为灰度图并缩放。处理后的图片会被记录到 [`rerun`](https://github.com/rerun-io/rerun) 流中。

更多用例请查阅 [`examples`](https://github.com/kornia/kornia-rs/tree/main/examples) 目录。

```rust
use kornia::{image::{Image, ImageSize}, imgproc};
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 读取图片
    let image: Image<u8, 3, _> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;
    let image_viz = image.clone();

    let image_f32: Image<f32, 3, _> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // 转为灰度图
    let mut gray = Image::<f32, 1, _>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // 缩放图片
    let new_size = ImageSize {
        width: 128,
        height: 128,
    };

    let mut gray_resized = Image::<f32, 1, _>::from_size_val(new_size, 0.0)?;
    imgproc::resize::resize_native(
        &gray, &mut gray_resized,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    println!("gray_resize: {:?}", gray_resized.size());

    // 创建 Rerun 记录流
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            image_viz.as_slice(),
            image_viz.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "gray",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "gray_resize",
        &rerun::Image::from_elements(
            gray_resized.as_slice(),
            gray_resized.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
```

![Screenshot from 2024-03-09 14-31-41](https://github.com/kornia/kornia-rs/assets/5157099/afdc11e6-eb36-4fcc-a6a1-e2240318958d)

## Python 用法

加载图片，并直接转换为 numpy 数组，方便与其他库集成。

```python
import kornia_rs as K
import numpy as np
import torch

# 使用 libjpeg-turbo 加载图片
img: np.ndarray = K.read_image_jpeg("dog.jpeg")

# 或者，加载其他格式
# img: np.ndarray = K.read_image_any("dog.png")

assert img.shape == (195, 258, 3)

# 转为 dlpack 以导入 torch
img_t = torch.from_dlpack(img)
assert img_t.shape == (195, 258, 3)
```

将图片写入磁盘

```python
import kornia_rs as K
import numpy as np

# 使用 libjpeg-turbo 加载图片
img: np.ndarray = K.read_image_jpeg("dog.jpeg")

# 写入图片到磁盘
K.write_image_jpeg("dog_copy.jpeg", img)
```

使用 `turbojpeg` 后端对图像流进行编码或解码

```python
import kornia_rs as K

# 用 kornia-rs 加载图片
img = K.read_image_jpeg("dog.jpeg")

# 用 jpeg 编码图片
image_encoder = K.ImageEncoder()
image_encoder.set_quality(95)  # 设置编码质量

# 获取编码后的流
img_encoded: list[int] = image_encoder.encode(img)

# 解码回图片
image_decoder = K.ImageDecoder()

decoded_img: np.ndarray = image_decoder.decode(bytes(img_encoded))
```

使用 `kornia-rs` 后端和 SIMD 加速缩放图片

```python
import kornia_rs as K

# 用 kornia-rs 加载图片
img = K.read_image_jpeg("dog.jpeg")

# 缩放图片
resized_img = K.resize(img, (128, 128), interpolation="bilinear")

assert resized_img.shape == (128, 128, 3)
```

## 🧑‍💻 开发

前置条件：系统中需安装 `rust` 和 `python3`。

安装 rustup
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装 [`uv`](https://docs.astral.sh/uv/) 以管理 python 依赖
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装 [`just`](https://github.com/casey/just) 命令行工具，用于管理开发任务。
```bash
cargo install just
```

克隆仓库到本地目录
```bash
git clone https://github.com/kornia/kornia-rs.git
```

你可以在项目根目录下运行 `just` 查看可用命令。

```bash
$ just
Available recipes:
    check-environment                 # 检查项目所需的二进制文件是否已安装
    clean                             # 清理缓存和构建产物
    clippy                            # 用所有特性运行 clippy
    clippy-default                    # 用默认特性运行 clippy
    fmt                               # 自动格式化和 lint
    py-build py_version='3.9'         # 创建虚拟环境并构建 kornia-py
    py-build-release py_version='3.9' # 创建虚拟环境并为发布构建 kornia-py
    py-install py_version='3.9'       # 创建虚拟环境并安装开发依赖
    py-test                           # 用 pytest 测试 kornia-py 代码
    test name=''                      # 测试全部或指定测试
```
### 🐳 Devcontainer

本项目包含开发容器，提供一致的开发环境。

开发容器已配置好所有构建和测试 `kornia-rs` 所需的依赖和工具，确保不同机器和环境下开发体验一致。

**使用方法**

1. **安装 Remote - Containers 扩展**：在 Visual Studio Code 中，从扩展视图（`Ctrl+Shift+X`）安装 `Remote - Containers` 扩展。

2. **在容器中打开项目**：
    - 在 Visual Studio Code 中打开 `kornia-rs` 项目文件夹。
    - 按 `F1` 并选择 `Remote-Containers: Reopen in Container`。

Visual Studio Code 会构建容器并在其中打开项目。你可以在容器环境中开发、构建和测试项目。

### 🦀 Rust

编译项目并运行测试

```bash
just test
```

如需运行指定测试，可用如下命令：

```bash
just test image
```

### 🐍 Python

构建 Python wheel 包，需使用 `maturin` 包。使用如下命令构建 wheel：

```bash
just py-build
```

运行测试：

```bash
just py-test
```

## 💜 贡献

本项目为 [Kornia](https://github.com/kornia/kornia) 的子项目。欢迎加入社区与我们交流，或通过 <https://opencollective.com/kornia> 赞助项目。

### AI 政策

Kornia-rs 接受 AI 辅助的代码，但严格拒绝提交者仅作为代理的 AI 生成贡献。所有贡献者必须是每一行代码的**唯一责任作者**。在提交 pull request 之前，请查看我们的 [AI 政策](AI_POLICY.md)。主要要求包括：

- **验证证明**：PR 必须包含本地测试日志以证明代码已执行（例如 `pixi run rust-test` 或 `cargo test`）
- **预先讨论**：所有 PR 在实施前必须在 Discord 或通过 GitHub issue 进行讨论
- **库引用**：实现必须基于现有库引用（Rust crates、OpenCV 等）
- **使用现有工具**：使用现有的 `kornia-rs` 工具，而不是重新发明轮子
- **错误处理**：使用 `Result<T, E>` 进行错误处理（在库代码中避免使用 `unwrap()`/`expect()`）
- **解释能力**：您必须能够解释您提交的任何代码

自动化 AI 审查工具（例如 @copilot）将根据这些政策检查 PR。完整详情请参阅 [AI_POLICY.md](AI_POLICY.md)。

## 论文引用

如果您在研究中使用了 kornia-rs，请引用：

```bibtex
@misc{2505.12425,
Author = {Edgar Riba and Jian Shi and Aditya Kumar and Andrew Shen and Gary Bradski},
Title = {Kornia-rs: A Low-Level 3D Computer Vision Library In Rust},
Year = {2025},
Eprint = {arXiv:2505.12425},
}
```
