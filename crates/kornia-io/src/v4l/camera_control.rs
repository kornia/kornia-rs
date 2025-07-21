/// Core trait that all camera controls must implement
pub trait CameraControlTrait: std::fmt::Debug + Send + Sync {
    /// Human-readable name of the control
    fn name(&self) -> &str;

    /// V4L2 control ID
    fn control_id(&self) -> u32;

    /// Current value as V4L2 compatible integer
    fn value(&self) -> ControlType;

    /// Control description
    fn description(&self) -> String;
}

/// Define the control type for the camera control
#[derive(Debug)]
pub enum ControlType {
    /// Integer control type
    Integer(i64),
    /// Boolean control type
    Boolean(bool),
}

/// The camera control for brightness
#[derive(Debug, Clone, PartialEq)]
pub struct Brightness(pub i32);

#[rustfmt::skip]
impl CameraControlTrait for Brightness {
    fn name(&self) -> &str { "brightness" }
    fn control_id(&self) -> u32 { 0x00980900 }
    fn value(&self) -> ControlType { ControlType::Integer(self.0 as i64) }
    fn description(&self) -> String { "Picture brightness control".to_string() }
}

/// The camera control for contrast
#[derive(Debug, Clone, PartialEq)]
pub struct Contrast(pub i32);

#[rustfmt::skip]
impl CameraControlTrait for Contrast {
    fn name(&self) -> &str { "contrast" }
    fn control_id(&self) -> u32 { 0x00980901 }
    fn value(&self) -> ControlType { ControlType::Integer(self.0 as i64) }
    fn description(&self) -> String { "Picture contrast control".to_string() }
}

/// The camera control for saturation
#[derive(Debug, Clone, PartialEq)]
pub struct Saturation(pub i32);

#[rustfmt::skip]
impl CameraControlTrait for Saturation {
    fn name(&self) -> &str { "saturation" }
    fn control_id(&self) -> u32 { 0x00980902 }
    fn value(&self) -> ControlType { ControlType::Integer(self.0 as i64) }
    fn description(&self) -> String { "Picture saturation control".to_string() }
}

/// The camera control for hue
#[derive(Debug, Clone, PartialEq)]
pub struct Hue(pub i32);

#[rustfmt::skip]
impl CameraControlTrait for Hue {
    fn name(&self) -> &str { "hue" }
    fn control_id(&self) -> u32 { 0x00980903 }
    fn value(&self) -> ControlType { ControlType::Integer(self.0 as i64) }
    fn description(&self) -> String { "Picture hue control".to_string() }
}

/// The camera control for sharpness
#[derive(Debug, Clone, PartialEq)]
pub struct Sharpness(pub i32);

#[rustfmt::skip]
impl CameraControlTrait for Sharpness {
    fn name(&self) -> &str { "sharpness" }
    fn control_id(&self) -> u32 { 0x0098091b }
    fn value(&self) -> ControlType { ControlType::Integer(self.0 as i64) }
    fn description(&self) -> String { "Picture sharpness control".to_string() }
}
