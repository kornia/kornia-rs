import kornia_rs as K

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
    kind = tf.into_family_kind()
    assert kind.name == "custom"
