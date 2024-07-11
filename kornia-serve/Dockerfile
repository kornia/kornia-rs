# Use an official Rust runtime as a parent image
FROM rust:1.79-bullseye as builder

# Set the working directory in the image to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    nasm

# Build the application in release mode
RUN cargo build --release

# Our production image starts here
FROM debian:buster-slim

# Copy the binary from the builder stage to the production image
COPY --from=builder /app/target/release/kornia-serve /usr/local/bin

# Expose port 3000
EXPOSE 3000

# Set the startup command to run your binary
CMD ["kornia-serve"]
