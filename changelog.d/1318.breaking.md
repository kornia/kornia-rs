`StreamCaptureError` has a new `StreamError(gstreamer::glib::Error)` variant, returned when the
capture pipeline posts an error on its bus (e.g. an undecodable file or a missing decoder) instead
of silently delivering no more frames. It carries GStreamer's error domain, code and debug
details, so a `match` on the enum needs an arm for it.
