import kornia_rs as K

TF = K.apriltag.family.TagFamilyKind


def test_all_tag_family_kind():
    all = K.apriltag.family.all_tag_family_kind()
    all_expected = [
        TF("tag16_h5"),
        TF("tag36_h11"),
        TF("tag36_h10"),
        TF("tag25_h9"),
        TF("tagcircle21_h7"),
        TF("tagcircle49_h12"),
        TF("tagcustom48_h12"),
        TF("tagstandard41_h12"),
        TF("tagstandard52_h13"),
    ]

    assert all == all_expected
