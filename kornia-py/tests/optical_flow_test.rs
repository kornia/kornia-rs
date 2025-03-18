use kornia::tensor::Tensor;
use tch::{Device, Kind};
use crate::pyramid::build_pyramid;
use crate::tracking::lucas_kanade;

#[test]
fn test_optical_flow() {
    let prev_img = Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu));
    let next_img = Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu));

    let points = vec![(32.0, 32.0), (64.0, 64.0)];
    
    let pyramid = build_pyramid(&prev_img, 3);
    assert_eq!(pyramid.len(), 3);

    let flow = lucas_kanade(&prev_img, &next_img, &points, 5);
    assert_eq!(flow.len(), points.len());
}
