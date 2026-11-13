/// An error type for the stream module.
#[derive(thiserror::Error, Debug)]
pub enum StreamCaptureError {
    /// A GStreamer error: initialization, building the pipeline, or an error posted by a
    /// pipeline element while it runs (with the element and debug details in the message).
    #[error(transparent)]
    GStreamerError(#[from] gstreamer::glib::Error),

    /// An error occurred during GStreamer downcast of pipeline element.
    #[error("Failed to downcast pipeline")]
    DowncastPipelineError(gstreamer::Element),

    /// An error occurred during GStreamer downcast of appsink.
    #[error("Failed to get an element by name")]
    GetElementByNameError,

    /// An error occurred during GStreamer to get the bus.
    #[error("Failed to get the bus")]
    BusError,

    /// An error occurred during GStreamer to set the pipeline state.
    #[error(transparent)]
    SetPipelineStateError(#[from] gstreamer::StateChangeError),

    /// An error occurred during GStreamer to pull sample from appsink.
    #[error(transparent)]
    PullSampleError(#[from] gstreamer::glib::BoolError),

    /// An error occurred during GStreamer to get the caps from the sample.
    #[error("Failed caps: {0}")]
    GetCapsError(String),

    /// An error occurred during GStreamer to get the buffer from the sample.
    #[error("Failed to get the buffer from the sample")]
    GetBufferError,

    /// An error occurred during GStreamer to map the buffer to an ImageFrame.
    #[error("Failed to create an image frame")]
    CreateImageFrameError,

    /// An error occurred during GStreamer to send eos event.
    #[error("Failed to send eos event")]
    SendEosError,

    /// An error occurred during GStreamer to send flush start event.
    #[error("Pipeline cancelled by the user")]
    PipelineCancelled,

    /// An error for an invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An error occurred during GStreamer to send end of stream event.
    #[error(transparent)]
    GstreamerFlowError(#[from] gstreamer::FlowError),

    /// An error occurred during checking the image format.
    #[error("Invalid image format: {0}")]
    InvalidImageFormat(String),

    /// An error occurred when the pipeline is not running.
    #[error("Pipeline is not running")]
    PipelineNotRunning,

    /// An error occurred when the allocator is not found.
    #[error("Could not lock the mutex")]
    MutexPoisonError,

    /// The GStreamer buffer is smaller than the frame layout it should hold.
    #[error("buffer size mismatch: expected {expected} bytes, got {got}")]
    BufferSizeMismatch {
        /// The minimum number of bytes the frame layout needs: the plane offset plus every
        /// row's pixels at the row stride (the padding after the last row is not required).
        expected: usize,
        /// The number of bytes in the GStreamer buffer.
        got: usize,
    },

    /// Building a kornia `Image` from a captured GStreamer frame failed.
    ///
    /// Returned when the frame's pixel data cannot form an image of the negotiated size, for
    /// example when the data length does not match `width * height * channels`.
    /// The wrapped [`kornia_image::ImageError`] says exactly what was wrong, and this error
    /// shows its message unchanged. Callers of the video reader receive it inside
    /// [`VideoReaderError::StreamCaptureError`].
    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    /// An error occurred when joining a thread.
    #[error("Failed to join thread")]
    JoinThreadError,

    /// The capture pipeline posted an error on its bus (e.g. a file that cannot be decoded or a
    /// missing decoder). The error carries GStreamer's domain, code, element and debug details.
    #[error("stream error: {0}")]
    StreamError(#[source] gstreamer::glib::Error),
}

/// Error type for video reader
#[derive(thiserror::Error, Debug)]
pub enum VideoReaderError {
    /// An error occurred while seeking within the video stream.
    #[error("Failed to seek")]
    SeekError,

    /// An error occurred while attempting to retrieve the current position in the video stream.
    #[error("Failed to get the current position in the video")]
    CurrentPosError,

    /// The specified playback speed is invalid. Playback speed must be greater than zero.
    #[error("The playback speed is invalid i.e. either it's negative or zero")]
    InvalidPlaybackSpeed,

    /// An error occurred in the stream capture process.
    #[error(transparent)]
    StreamCaptureError(#[from] StreamCaptureError),
}

#[cfg(test)]
mod tests {
    use super::{StreamCaptureError, VideoReaderError};
    use kornia_image::{Image, ImageError, ImageSize};

    /// Builds a 2x2 RGB image; `?` converts an `ImageError` into a `StreamCaptureError`.
    fn image_from_bytes(data: Vec<u8>) -> Result<Image<u8, 3>, StreamCaptureError> {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        Ok(Image::<u8, 3>::new(size, data)?)
    }

    #[test]
    fn image_error_reaches_video_reader_error_unchanged() {
        // A 2x2 RGB image needs 2 * 2 * 3 = 12 bytes; 5 bytes must fail.
        let Err(err) = image_from_bytes(vec![0u8; 5]) else {
            panic!("expected an error for 5 bytes of a 2x2 RGB image");
        };

        assert!(matches!(
            err,
            StreamCaptureError::ImageError(ImageError::InvalidChannelShape(5, 12))
        ));

        let expected = ImageError::InvalidChannelShape(5, 12).to_string();
        assert_eq!(err.to_string(), expected);

        let reader_err = VideoReaderError::from(err);
        assert!(matches!(
            reader_err,
            VideoReaderError::StreamCaptureError(StreamCaptureError::ImageError(_))
        ));
        assert_eq!(reader_err.to_string(), expected);
    }
}
