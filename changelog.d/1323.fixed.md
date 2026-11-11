`VideoReader` and `StreamCapture` can now report end-of-stream and pipeline errors instead of
leaving callers to invent timeouts. `try_grab_rgb8`/`try_grab_gray8` return `Grab::Eos` or an
error; the legacy `grab_rgb8`/`grab_gray8` still return `Ok(None)` when no frame is buffered.
End-of-stream is read from the GStreamer bus, so it no longer depends on the position reaching
the duration or on a timeout, and a file that cannot decode returns an error instead of hanging.
