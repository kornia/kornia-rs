# kornia-rs: Low level implementations for Computer Vision in Rust.

[![Continuous integration](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ)

This project provides low level functionality for Computer Vision written in [Rust](https://www.rust-lang.org/) to be consumed by machine learning and data-science frameworks, specially those working with images. We mainly aim to provide I/O functionality for images (future: video, cameras), and visualisation in future.

- The library is written in [Rust](https://www.rust-lang.org/).
- Python bindings are created with [PyO3/Maturin](https://github.com/PyO3/maturin).
- We package with support for Linux [amd64/arm64], Macos and WIndows.
- Supported Python versions are 3.7/3.8/3.9/3.10/3.11

## Basic Usage

Load an image, that is converted to `cv::Tensor` wich is a centric structure to the DLPack protocol to share tensor data across frameworks with a zero-copy cost.

```python
    import kornia_rs as K
    from kornia_rs import Tensor as cvTensor

    # load an image with Rust `image-rs` as backend library
    cv_tensor: cvTensor = K.read_image_rs("dog.jpeg")
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)
    assert th_tensor.shape == (195, 258, 3)
    assert np_tensor.shape == (195, 258, 3)

    # or to numpy with same interface
    np_tensor = np.from_dlpack(cv_tensor)
```

## Advanced usage

Encode or decoda image streams using the `turbojpeg` backend

```python
# load image using turbojpeg
cv_tensor = K.read_image_jpeg("dog.jpeg")
image: np.ndarray = np.from_dlpack(cv_tensor)  # HxWx3

# encode the image with jpeg
image_encoder = K.ImageEncoder()
image_encoder.set_quality(95)  # set the encoding quality

# get the encoded stream
image_encoded: List[int] = image_encoder.encode(image.tobytes(), image.shape)

# write to disk the encoded stream
K.write_image_jpeg("dog_encoded.jpeg", image_encoded)

# decode back the image
image_decoder = K.ImageDecoder()

decoded_tensor = image_decoder.decode(bytes(image_encoded))
decoded_image: np.ndarray = np.from_dlpack(decoded_tensor)  # HxWx3
```

## TODO: short/mid-terrm

- [x] [infra] Automate packaging for manywheels.
- [x] [kornia] integrate with the new `Image` API
- [x] [dlpack] move dlpack implementation to dlpack-rs.
- [x] [dlpack] implement test for torch and numpy.
- [ ] [dlpack] update dlpack version >=0.8
- [ ] [dlpack] implement `DLPack` to `cv::Tensor`.

## TODO: not priority for now

- [ ] [io] Implement image encoding and explore video.
- [ ] [viz] Fix minor issues and implement a full `VizManager` to work on the browser.
- [ ] [tensor] implement basic functionality to test: add, sub, mul, etc.
- [ ] [tensor] explore xnnpack and openvino integration.

## Development

To test the project in lyour local machine use the following instructions:

1. Clone the repository in your local directory

```bash
git clone https://github.com/kornia/kornia-rs.git
```

2.1 (optional) Build the `devel.Dockerfile`

Let's prepare the development environment with Docker.
Make sure you have docker in your system: https://docs.docker.com/engine/install/ubuntu/

```bash
cd ./docker && ./build_devel.sh
KORNIA_RS_DEVEL_IMAGE="kornia_rs/devel:local" ./devel.sh
```

2.2 Enter to the `devel` docker container.

```bash
./devel.sh
```

3. Build the project

(you should now be inside the docker container)

```bash
# maturin needs you to be a `venv`
python3 -m venv .venv
source .venv/bin/activate

# build and generate linked wheels
maturin develop --extras dev
```

4. Run the tests

```bash
pytest test/
```

## Contributing

This is a child project of [Kornia](https://github.com/kornia/kornia). Join the community to get in touch with us, or just sponsor the project: https://opencollective.com/kornia
