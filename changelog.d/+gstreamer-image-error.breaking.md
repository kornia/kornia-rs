`StreamCaptureError::ProcessImageFrameError` is removed: nothing in kornia-io returned it, and
failures building an `Image` from a GStreamer frame are reported as `StreamCaptureError::ImageError`.
A `match` arm for the removed variant must go, and `?` no longer converts a
`Box<dyn Error + Send + Sync>` into a `StreamCaptureError`.