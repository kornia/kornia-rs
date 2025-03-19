@_default:
  just --list

# ------------------------------------------------------------------------------
# Recipes for the kornia rust project
# ------------------------------------------------------------------------------

# Check if the code is linting and formatted
check:
  @cargo check --workspace --all-targets --all-features --locked

# Check if the required binaries for the project are installed
check-environment:
  @echo "Rust version." && cargo --version

# Run clippy with all features
clippy:
  @echo "🚀 Running clippy"
  cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

# Run clippy with default features
clippy-default:
  @echo "🚀 Running clippy"
  cargo clippy --all-targets --locked -- -D warnings

# Run autoformatting and linting
format:
  cargo fmt --all

# Clean up caches and build artifacts
clean:
  @rm -rf .venv/
  @rm -rf target/
  @rm -f Cargo.lock
  @cargo clean

# Test the code or a specific test
test name='':
  @cargo test {{ name }}

# Test the code with all features
test-all:
  @cargo test --all-features

# ------------------------------------------------------------------------------
# Recipes for the kornia-py project
# ------------------------------------------------------------------------------

# Create virtual environment, and install dev requirements
py-install py_version='3.9':
  @cd kornia-py/ && just install {{ py_version }}

# Create virtual environment, and build kornia-py
py-build py_version='3.9':
  @cd kornia-py/ && just build {{ py_version }}

# Create virtual environment, and build kornia-py for release
py-build-release py_version='3.9':
  @cd kornia-py/ && just build {{ py_version }} "--release"

# Test the kornia-py code with pytest
py-test:
  @cd kornia-py/ && just test
