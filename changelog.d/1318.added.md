- `StreamCapture::try_grab_rgb8` / `try_grab_gray8` and `VideoReader::try_grab_rgb8` /
  `try_grab_gray8`, returning `Grab::{Frame, Pending, Eos}` so "no frame yet", end-of-stream and a
  pipeline error cannot be confused.
- `StreamCapture::is_eos` / `VideoReader::is_eos` and `stream_error`, to inspect the terminal
  stream state without grabbing a frame.
- `StreamCaptureError::StreamError` carrying the GStreamer error posted on the pipeline bus.
- `VideoReader::set_realtime`, to release frames at their presentation time for paced playback
  (decoding otherwise runs as fast as possible).
