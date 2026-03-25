An example showing the use of contours module which uses Suzuki-Abe algorithm to find contour lines.

NOTE: Build release version first

```bash
cargo build --release
```

```bash
Usage: cargo run --release -- --image <IMAGE_PATH> --approx <simple | none> --threshold 60
```

Output example:

| Depth | Colour | Meaning |
|-------|--------|---------|
| 0 | Green | Outermost contours |
| 1 | Orange | First-level holes |
| 2 | Blue | Nested inside depth-1 |
| 3 | Yellow | Nested inside depth-2 |
| 4 | Red | Nested inside depth-3 |

![](https://github.com/user-attachments/assets/6e5bd62c-00ea-4f3c-be14-32ec6e79ba4b)
