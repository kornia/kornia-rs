`VideoReader` and `StreamCapture` now report the end of a stream and pipeline errors instead of
returning `Ok(None)` indefinitely from `grab_rgb8` / `grab_gray8`. End-of-stream is read from the
GStreamer bus, so callers no longer need position/duration or timeout heuristics, and files that can
never decode (audio-only, truncated, non-media, missing decoder) return an error instead of hanging.
File decoding no longer waits for each frame's presentation time, so offline reads finish as fast as
the decoder allows.
