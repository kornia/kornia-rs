Example using kornia-rs and dora-rs to build a simple pipeline to process images from multiple sources.

Instructions:

1. Install dependencies:

```bash
cargo install dora-cli
```

2. Build the pipeline:

```bash
dora build dataflow.yml
```

3. Run the pipeline:

```bash
dora run dataflow.yml
```

4. Open the rerun viewer:

```bash
rerun --bind 127.0.0.1 --port 9876
```
![Screenshot from 2025-03-18 16-37-20](https://github.com/user-attachments/assets/e1372851-7ba3-4890-a98e-703855cfe304)
