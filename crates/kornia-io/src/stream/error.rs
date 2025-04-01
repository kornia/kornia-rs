/// An error type for the stream module.
#[derive(thiserror::Error, Debug)]
pub enum StreamCaptureError {
    /// An error occurred during GStreamer initialization.
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

    // TODO: support later on ImageError
    /// An error occurred during processing the image frame.
    #[error(transparent)]
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
}
