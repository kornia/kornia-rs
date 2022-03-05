import numpy as np

import torch
from kornia.testing import assert_close
from kornia_io import Image, ImageColor
from kornia.color.gray import rgb_to_grayscale

import kornia_io
from kornia_io.core.image import read_image


class TestImage:
    def test_smoke(self):
        data = torch.rand(3, 4, 5)
        img = Image.from_tensor(data, color=ImageColor.RGB)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB
        assert not img.is_batch

    def test_from_numpy(self):
        data = np.ones((4, 5, 3))
        img = Image.from_numpy(data, color=ImageColor.RGB)
        assert img.channels == 3
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.RGB
        assert not img.is_batch

    def test_from_list(self):
        data = [[[1., 2.], [3., 4.]]]
        img = Image.from_list(data, color=ImageColor.GRAY)
        assert img.channels == 1
        assert img.height == 2
        assert img.width == 2
        assert img.color == ImageColor.GRAY
        assert not img.is_batch

    def test_grayscale(self):
        data = torch.rand(3, 4, 5)
        img = Image.from_tensor(data, color=ImageColor.RGB)
        img_gray = img.grayscale()
        assert img_gray.channels == 1
        assert img_gray.height == 4
        assert img_gray.width == 5
        assert img_gray.color == ImageColor.GRAY
        assert not img.is_batch
        assert_close(img_gray, rgb_to_grayscale(img))
        # inplace
        img = img.grayscale_()
        assert img.channels == 1
        assert img.height == 4
        assert img.width == 5
        assert img.color == ImageColor.GRAY
        assert not img.is_batch
        assert_close(img_gray, img)

    def test_from_file(self):
        img_file = '/home/edgar/Downloads/IMG_20211219_145924.jpg' 
        img = Image.from_file(img_file)
        assert img.channels == 3
        assert img.height == 600
        assert img.width == 800
        assert img.color == ImageColor.RGB
        assert not img.is_batch

    def test_read_image(self):
        img_file = '/home/edgar/Downloads/IMG_20211219_145924.jpg' 
        img: Image = read_image(img_file)
        assert img.channels == 3
        assert img.height == 600
        assert img.width == 800
        assert img.color == ImageColor.RGB
        assert not img.is_batch