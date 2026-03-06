Example showing EXIF-aware JPEG decode in `kornia-io`.

It writes:
- raw JPEG decode (no EXIF orientation correction)
- EXIF auto-oriented decode

## Usage

```bash
cargo run -p exif_auto_orient -- <input.jpg> [raw_out.jpg] [fixed_out.jpg]
```

Example:

```bash
cargo run -p exif_auto_orient -- \
  /path/to/landscape_3.jpg \
  /tmp/exif_before_raw.jpg \
  /tmp/exif_after_auto_orient.jpg
```

Expected behavior:
- `read_image_jpeg_rgb8` returns raw decoded pixels.
- `read_image_jpeg_auto_orient` reads EXIF Orientation and remaps pixels to the physically correct pose.
