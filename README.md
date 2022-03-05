# kornia_rs
Low level implementations for computer vision in Rust

## (🚨 Warning: Unstable Prototype 🚨)

## Development

To test the project use the following instructions:

1. Clone the repository in your local directory

```bash
git clone https://github.com/kornia/kornia_rs.git
```

2. Build the `devel.Dockerfile`

Let's prepare the development environment with Docker.
Make sure you have docker in your system: https://docs.docker.com/engine/install/ubuntu/

```bash
cd kornia_rs/docker
./build_devel.sh && cd ../ 
./docker/devel.sh
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