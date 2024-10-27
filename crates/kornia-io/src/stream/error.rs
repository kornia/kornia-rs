/// An error type for the stream module.
#[derive(thiserror::Error, Debug)]
pub enum StreamCaptureError {
    /// An error occurred during GStreamer initialization.
    #[error("Failed to initialize GStreamer")]
    GStreamerError(#[from] gst::glib::Error),

    /// An error occurred during GStreamer downcast of pipeline element.
    #[error("Failed to downcast pipeline")]
    DowncastPipelineError(gst::Element),

    /// An error occurred during GStreamer downcast of appsink.
    #[error("Failed to get an element by name")]
    GetElementByNameError,

    /// An error occurred during GStreamer to get the bus.
    #[error("Failed to get the bus")]
    BusError,

    /// An error occurred during GStreamer to set the pipeline state.
    #[error("Failed to set the pipeline state")]
    SetPipelineStateError(#[from] gst::StateChangeError),

    /// An error occurred during GStreamer to pull sample from appsink.
    #[error("Failed to pull sample from appsink")]
    PullSampleError(#[from] gst::glib::BoolError),

    /// An error occurred during GStreamer to get the caps from the sample.
    #[error("Failed to get the caps from the sample")]
    GetCapsError,

    /// An error occurred during GStreamer to get the structure from the caps.
    #[error("Failed to get the structure")]
    GetStructureError,

    // TODO: figure out the #[from] macro for this error
    /// An error occurred during GStreamer to get the height from the structure.
    #[error("Failed to get the height from the structure")]
    GetHeightError,

    // TODO: figure out the #[from] macro for this error
    /// An error occurred during GStreamer to get the width from the structure.
    #[error("Failed to get the width from the structure")]
    GetWidthError,

    /// An error occurred during GStreamer to get the buffer from the sample.
    #[error("Failed to get the buffer from the sample")]
    GetBufferError,

    /// An error occurred during GStreamer to map the buffer to an ImageFrame.
    #[error("Failed to create an image frame")]
    CreateImageFrameError,

    // TODO: support later on ImageError
    /// An error occurred during processing the image frame.
    #[error("Failed processing the image frame")]
    ProcessImageFrameError(#[from] Box<dyn std::error::Error + Send + Sync>),

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
    #[error("Error ocurred in the gstreamer flow")]
    GstreamerFlowError(#[from] gst::FlowError),

    /// An error occurred during checking the image format.
    #[error("Invalid image format: {0}")]
    InvalidImageFormat(String),

    /// An error occurred when the pipeline is not running.
    #[error("Pipeline is not running")]
    PipelineNotRunning,
}

// ensure that can be sent over threads
unsafe impl Send for StreamCaptureError {}
unsafe impl Sync for StreamCaptureError {}
