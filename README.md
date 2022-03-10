# kornia-rs
Low level implementations for computer vision in Rust

## (🚨 Warning: Unstable Prototype 🚨)

[![Continuous integration](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/kornia-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ)

## Development

To test the project use the following instructions:

1. Clone the repository in your local directory

```bash
git clone https://github.com/kornia/kornia-rs.git
```

2. Build the `devel.Dockerfile`

Let's prepare the development environment with Docker.
Make sure you have docker in your system: https://docs.docker.com/engine/install/ubuntu/

```bash
cd ./docker && ./build_devel.sh
cd ../ && ./devel.sh
```

3. Build the project

(you should be inside the docker container)

```bash
pip3 install -e .[dev]
```

4. Run the tests

```bash
pytest test/
```