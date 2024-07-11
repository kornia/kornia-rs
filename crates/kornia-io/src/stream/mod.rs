mod camera;
pub use camera::{CameraCapture, CameraCaptureBuilder};

/// An error type for WebcamCapture
#[derive(thiserror::Error, Debug)]
pub enum StreamCaptureError {
    #[error("Failed to initialize GStreamer")]
    GStreamerError(#[from] gst::glib::Error),

    #[error("Failed to downcast pipeline")]
    DowncastPipelineError(gst::Element),

    #[error("Failed to downcast appsink")]
    DowncastAppSinkError,

    #[error("Failed to get the bus")]
    BusError,

    #[error("Failed to set the pipeline state")]
    SetPipelineStateError(#[from] gst::StateChangeError),

    #[error("Failed to pull sample from appsink")]
    PullSampleError(#[from] gst::glib::BoolError),

    #[error("Failed to get the caps from the sample")]
    GetCapsError,

    #[error("Failed to get the structure")]
    GetStructureError,

    // TODO: figure out the #[from] macro for this error
    #[error("Failed to get the height from the structure")]
    GetHeightError,

    // TODO: figure out the #[from] macro for this error
    #[error("Failed to get the width from the structure")]
    GetWidthError,

    #[error("Failed to get the buffer from the sample")]
    GetBufferError,

    #[error("Failed to create an image frame")]
    CreateImageFrameError,

    // TODO: support later on ImageError
    #[error("Failed processing the image frame")]
    ProcessImageFrameError(#[from] Box<dyn std::error::Error>),

    #[error("Failed to send eos event")]
    SendEosError,

    #[error("Pipeline cancelled by the user")]
    PipelineCancelled,
}
