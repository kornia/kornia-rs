`StreamCaptureError` is now `#[non_exhaustive]` and has a new `StreamError(gstreamer::glib::Error)`
variant, returned when the capture pipeline posts an error on its bus (e.g. an undecodable file or
a missing decoder) instead of silently delivering no more frames. Downstream `match`es need a `_`
arm; the variant carries GStreamer's error domain, code and debug details.
