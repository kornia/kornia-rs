An example showcasing how to write a video file reader using `kornia_io` module along
with wecam capture and recording example. Visualizes the webcam and video feed in a
[rerun](https://github.com/rerun-io/rerun) window.

```
Usage: video_read [-c <camera-id>] [-d <duration>] -o <output> [-w <width>] [-h <height>] [-f <fps>]

Read the video recorded from webcam.

Options:
  -c, --camera-id   the camera id to use
  -d, --duration    duration of webcam stream to record
  -o, --output      path to file where webcam recording will be saved
  -w, --width       width of the video
  -h, --height      height of the video
  -f, --fps         the frames per second of video3
  --help, help      display usage information
```

Example:

```bash
cargo run -p video_read --release -- --output ~/output.mp4
```
