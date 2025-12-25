# Kornia Image API Examples

Comprehensive examples demonstrating the kornia-image Rust API.

## Running Examples

```bash
# From the workspace root
cd examples/image_api
cargo run

# Or use cargo from anywhere
cargo run --manifest-path examples/image_api/Cargo.toml
```

## Example: Image API

The `image_api` example demonstrates:

1. **Creating images from fill values**
2. **Creating images from data vectors**
3. **Error handling with Result types**
4. **Zero-copy data access**
5. **Creating owned copies**
6. **Different image types** (u8, f32) and channels (1, 3, 4)
7. **Pixel-level access and manipulation**
8. **Creating images from slices**
9. **Channel extraction and splitting**

### Example Output

```
╔════════════════════════════════════════════╗
║   Kornia Rust Image API Examples          ║
╚════════════════════════════════════════════╝

Example 1: Creating images from fill values
================================================
✓ Created 640x480x3 image
  Fill value: 128
  Total bytes: 921600

Example 2: Creating images from data vectors
================================================
✓ Created 10x10 grayscale image from data
  First pixel: 0
  Last pixel: 252

Example 3: Error handling (wrong data size)
================================================
✓ Caught expected error:
  Data length (100) does not match the image size (300)
...
```

## API Features Demonstrated

### 1. Creating Images from Fill Values

```rust
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};

let size = ImageSize { width: 640, height: 480 };
let alloc = CpuAllocator::default();
let img = Image::<u8, 3, _>::from_size_val(size, 128, alloc)?;
```

### 2. Creating Images from Data Vectors

```rust
let data = vec![0u8; 100];
let size = ImageSize { width: 10, height: 10 };
let alloc = CpuAllocator::default();
let img = Image::<u8, 1, _>::new(size, data, alloc)?;
```

### 3. Zero-Copy Data Access

```rust
let img = Image::<u8, 3, _>::from_size_val(size, 42, alloc)?;
let data = img.as_slice();  // &[u8] - zero copy!
let pixel = data[0];
```

### 4. Owned Copy of Data

```rust
let img = Image::<f32, 3, _>::from_size_val(size, 0.5, alloc)?;
let owned_copy = img.to_vec();  // Vec<f32>
// Can modify independently of original
```

### 5. Error Handling

```rust
let data = vec![0u8; 100];  // Wrong size
let size = ImageSize { width: 10, height: 10 };
let alloc = CpuAllocator::default();

match Image::<u8, 3, _>::new(size, data, alloc) {
    Ok(img) => println!("Created image"),
    Err(ImageError::InvalidChannelShape(got, expected)) => {
        println!("Error: got {} bytes, expected {}", got, expected);
    }
    Err(e) => println!("Other error: {}", e),
}
```

### 6. Pixel-Level Access

```rust
let mut img = Image::<u8, 3, _>::from_size_val(size, 0, alloc)?;

// Set individual pixels
img.set_pixel(0, 0, 0, 255)?;  // Red channel at (0, 0)
img.set_pixel(0, 0, 1, 128)?;  // Green channel at (0, 0)
img.set_pixel(0, 0, 2, 64)?;   // Blue channel at (0, 0)

// Get individual pixels
let r = img.get_pixel(0, 0, 0)?;
let g = img.get_pixel(0, 0, 1)?;
let b = img.get_pixel(0, 0, 2)?;
```

### 7. Channel Operations

```rust
let img = Image::<u8, 3, _>::from_size_val(size, 128, alloc)?;

// Extract single channel
let red_channel = img.channel(0)?;  // Returns Image<u8, 1, _>

// Split into all channels
let channels = img.split_channels()?;  // Vec<Image<u8, 1, _>>
for ch in channels {
    println!("Channel data: {:?}", ch.as_slice());
}
```

### 8. Image Properties

```rust
let img = Image::<u8, 3, _>::from_size_val(size, 0, alloc)?;

let width = img.width();      // usize
let height = img.height();    // usize
let rows = img.rows();        // usize (same as height)
let cols = img.cols();        // usize (same as width)
let size = img.size();        // ImageSize
let num_ch = img.num_channels();  // Returns 3
```

## Supported Image Types

The `Image<T, C, A>` type is generic over:
- `T`: Data type (u8, u16, f32, f64, etc.)
- `C`: Number of channels (1, 3, 4, etc.)
- `A`: Allocator (typically `CpuAllocator`)

| Type | Description | Example |
|------|-------------|---------|
| `Image<u8, 1, A>` | Grayscale 8-bit | `Image::<u8, 1, _>::from_size_val(...)` |
| `Image<u8, 3, A>` | RGB 8-bit | `Image::<u8, 3, _>::from_size_val(...)` |
| `Image<u8, 4, A>` | RGBA 8-bit | `Image::<u8, 4, _>::from_size_val(...)` |
| `Image<f32, 1, A>` | Grayscale float | `Image::<f32, 1, _>::from_size_val(...)` |
| `Image<f32, 3, A>` | RGB float | `Image::<f32, 3, _>::from_size_val(...)` |
| `Image<f32, 4, A>` | RGBA float | `Image::<f32, 4, _>::from_size_val(...)` |

## Memory Management

- **Ownership**: Images own their data via `Tensor3`
- **Zero-copy**: `as_slice()` returns `&[T]` reference to internal data
- **Allocators**: Support custom allocators (default: `CpuAllocator`)
- **RAII**: Automatic cleanup when image goes out of scope

## Data Layout

Images use **row-major interleaved** format:

```
For a 3x2 RGB image:
[R G B] [R G B] [R G B]  <- Row 0
[R G B] [R G B] [R G B]  <- Row 1

Pixel at (row=1, col=2): data[(1 * width + 2) * channels]
```

## Error Types

The `ImageError` enum provides structured error handling:

```rust
pub enum ImageError {
    /// Data length doesn't match expected size
    InvalidChannelShape(usize, usize),

    /// Invalid image dimensions
    InvalidImageSize(usize, usize, usize, usize),

    /// Pixel coordinates out of bounds
    PixelIndexOutOfBounds(usize, usize, usize, usize),

    /// Channel index out of bounds
    ChannelIndexOutOfBounds(usize, usize),

    // ... and more
}
```

All errors implement `std::error::Error` and have descriptive Display messages.

## Performance

- Zero-copy data access: O(1)
- Image creation: O(n) where n = width × height × channels
- Move operations: O(1) (ownership transfer)
- Clone operations: O(n) (deep copy of data)
- Pixel access: O(1) with bounds checking

## See Also

- [kornia-image crate documentation](https://docs.rs/kornia-image)
- [kornia-tensor crate](../kornia-tensor)
- [Main project README](../../README.md)
