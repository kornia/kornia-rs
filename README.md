# kornia-rs: Low level implementations for Computer Vision in Rust.

## (🚨 Warning: Unstable Prototype 🚨)

[![Continuous integration](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ)

The expectation of this project is to provide low level functionality
for Computer Vision written in [Rust](https://www.rust-lang.org/) to be consumed by deep learning frameworks, specially those working with images. We mainly provide I/O for images (future: video, cameras) and visualisation.

The library is written in [Rust](https://www.rust-lang.org/) and wrapped to Python (potentially later to C/C++) using [PyO3/Maturin](https://github.com/PyO3/maturin). The library can also be used as a standalone Rust crate.

## Basic Usage

Load an image, that is converted to `cv::Tensor` wich is designed centric
to the DLPack protocol to share tensor data across deep learning frameworks with a zero-copy cost.

The visualisation API is based on `vviz`: https://github.com/strasdat/vviz

```python
    import kornia_rs as K
    from kornia_rs import Tensor as cvTensor

    cv_tensor: cvTensor = K.read_image_rs("dog.jpeg")
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch and numpy
    # NOTE: later we will support to jax and mxnet.
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)
    np_tensor = np.from_dlpack(cv_tensor)
    assert th_tensor.shape == (195, 258, 3)
    assert np_tensor.shape == (195, 258, 3)
```

## TODO

- [x] [infra] Automate packaging for manywheels.
- [x] [kornia] integrate with the new `Image` API
- [x] [dlpack] move dlpack implementation to dlpack-rs.
- [ ] [dlpack] implement test for numpy, jax and mxnet.
- [ ] [dlpack] implement `DLPack` to `cv::Tensor`.
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
