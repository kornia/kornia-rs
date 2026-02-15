# Contributing to Kornia-py

## Setting Up the Environment

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/kornia-rs.git
    cd kornia-rs/kornia-py
    ```

2. **Set up the environment and install dependencies:**
    Install global dependencies necessary for building kornia rust package.
    ```sh
    sudo apt-get update --fix-missing \
    && sudo apt-get install -y --no-install-recommends \
                            cmake \
                            nasm \
                            libgstreamer1.0-dev \
                            libgstreamer-plugins-base1.0-dev \
    && sudo apt-get clean
    ```

    Or you can use one of the dev docker images ([x86_64](../devel-x86_64.Dockerfile), [i686](../devel-i686.Dockerfile), [arch64](../devel-aarch64.Dockerfile)).

    Then build the kornia-py package (from the project root):
    ```sh
    pixi run py-build
    ```

    Note:
    - When building the kornia-py package, it will automatically build its backend, which is the kornia Rust package itself.


3. **Run tests to ensure everything is set up correctly:**
    ```sh
    pixi run py-test
    ```

You are now ready to start contributing to the Kornia-py package!
