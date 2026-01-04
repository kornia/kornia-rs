from pathlib import Path

import kornia_rs as K
import numpy as np
import pytest

TAG36H11_TAG = (
    Path(__file__).parents[2]
    / "tests"
    / "data"
    / "apriltag-imgs"
    / "tag36h11"
    / "tag36_11_00005.png"
)

TagFamilyKind = K.apriltag.family.TagFamilyKind


def test_all_tag_family_kind():
    all = TagFamilyKind.all()
    all_expected = [
        TagFamilyKind("tag16_h5"),
        TagFamilyKind("tag36_h11"),
        TagFamilyKind("tag36_h10"),
        TagFamilyKind("tag25_h9"),
        TagFamilyKind("tagcircle21_h7"),
        TagFamilyKind("tagcircle49_h12"),
        TagFamilyKind("tagcustom48_h12"),
        TagFamilyKind("tagstandard41_h12"),
        TagFamilyKind("tagstandard52_h13"),
    ]

    assert all == all_expected


def test_tag_family_into_family_kind():
    qd = K.apriltag.family.QuickDecode(4, [1, 2, 3, 4])
    sb = K.apriltag.family.SharpeningBuffer(10)
    tf = K.apriltag.family.TagFamily(
        name="custom",
        width_at_border=2,
        reversed_border=False,
        total_width=10,
        nbits=4,
        bit_x=[0, 1, 2, 3],
        bit_y=[0, 1, 2, 3],
        code_data=[1, 2, 3, 4],
        quick_decode=qd,
        sharpening_buffer=sb,
    )
    kind = tf.into_tag_family_kind()
    assert kind.name == "custom"


def test_apriltag_decoder():
    kinds = [TagFamilyKind("tag36_h11")]
    config = K.apriltag.DecodeTagsConfig(kinds)
    decoder = K.apriltag.AprilTagDecoder(config, K.image.ImageSize(60, 60))

    expected_quad = [(50.0, 10.0), (50.0, 50.0), (10.0, 50.0), (10.0, 10.0)]

    with open(TAG36H11_TAG, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.io.decode_image_png_u8(bytes(img_data), (60, 60), "mono")
    assert img.shape == (60, 60, 1)
    assert img.dtype == np.uint8

    detection = decoder.decode(img)
    assert len(detection) == 1
    assert detection[0].id == 5

    for (ax, ay), (ex, ey) in zip(detection[0].quad.corners, expected_quad):
        assert ax == pytest.approx(ex, abs=1e-3)
        assert ay == pytest.approx(ey, abs=1e-3)
