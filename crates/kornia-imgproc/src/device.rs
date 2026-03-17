/// Supported devices for kornia hardware acceleration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KorniaDevice {
    /// CPU backend (default, Rayon parallelized where applicable).
    #[default]
    Cpu,
    /// GPU backend using CubeCL, with optional device index.
    Gpu(usize),
}
