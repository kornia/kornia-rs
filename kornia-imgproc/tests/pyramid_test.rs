#[test]
fn test_gaussian_pyramid() {
    use kornia::tensor::Tensor;
    use kornia_imgproc::pyramid::build_pyramid;

    let image = Tensor::randn(&[1, 3, 256, 256], (tch::Kind::Float, tch::Device::Cpu));
    
    let pyramid = build_pyramid(&image, 4);

    assert_eq!(pyramid.len(), 4);
    assert!(pyramid[1].size()[2] < pyramid[0].size()[2]);
}
