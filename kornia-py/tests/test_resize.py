from pathlib import Path
import kornia_rs as K
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def test_resize():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.io.read_image_jpeg(str(img_path.absolute()), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)

    img_resized: np.ndarray = K.imgproc.resize(img, (43, 34), "bilinear")
    assert img_resized.shape == (43, 34, 3)


def test_image_resize_nearest_non_rgb_matches_reference():
    """`Image.resize` on non-3-channel u8 images uses the generic nearest path;
    it must match `src[min(y*src_h//dst_h, src_h-1), min(x*src_w//dst_w, src_w-1)]`."""
    from kornia_rs.image import Image

    rng = np.random.default_rng(0)
    for (sh, sw, c), (dh, dw) in [
        ((5, 7, 1), (13, 3)),
        ((7, 5, 4), (2, 11)),
        ((1, 1, 2), (3, 5)),
        ((9, 17, 4), (9, 16)),
        ((16, 9, 1), (33, 31)),
    ]:
        src = rng.integers(0, 256, (sh, sw, c), dtype=np.uint8)
        got = Image(src).resize(dw, dh).numpy()
        ys = np.minimum(np.arange(dh) * sh // dh, sh - 1)
        xs = np.minimum(np.arange(dw) * sw // dw, sw - 1)
        want = src[ys][:, xs]
        assert got.shape == (dh, dw, c)
        np.testing.assert_array_equal(got, want)
