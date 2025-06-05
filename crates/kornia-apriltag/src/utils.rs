/// Trait representing a pixel value
pub trait PixelTrait {
    /// The black pixel value.
    const BLACK: Self;
    /// The white pixel value.
    const WHITE: Self;
    /// The pixel value used to indicate skipping processing.
    const SKIP_PROCESSING: Self;
}

macro_rules! impl_pixel_trait {
    ($($t:ty),*) => {
        $(
            impl PixelTrait for $t {
                const BLACK: Self = <$t>::MIN;
                const WHITE: Self = <$t>::MAX;
                const SKIP_PROCESSING: Self = 127;
            }
        )*
    };
}

impl_pixel_trait!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl PixelTrait for f32 {
    const BLACK: Self = 0.;
    const WHITE: Self = 1.;
    const SKIP_PROCESSING: Self = 0.127;
}

impl PixelTrait for f64 {
    const BLACK: Self = 0.;
    const WHITE: Self = 1.;
    const SKIP_PROCESSING: Self = 0.127;
}
