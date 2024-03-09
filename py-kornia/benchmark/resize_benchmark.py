from PIL import Image
import time
import numpy as np
import cv2

size = (256, 224)
new_size = (size[0] // 2, size[1] // 2)

image = np.ones((size[0], size[1], 3), dtype=np.uint8)

img_pil = Image.fromarray(image)

num_iters = 10000

times = []

for _ in range(num_iters):
    start = time.time()
    img_pil.resize(new_size, Image.BILINEAR)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)

print(f"RESIZE PIL: Average time: {np.median(times)}")


# test opencv resize

times = []

for _ in range(num_iters):
    start = time.time()
    cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    end = time.time()
    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)

print(f"RESIZE OPENCV: Average time: {np.median(times)}")


# test kornia resize

import torch
import kornia

image_torch = torch.from_numpy(image).permute(2, 0, 1).float() / 255.

times = []

for _ in range(num_iters):
    start = time.time()
    with torch.no_grad():
        kornia.geometry.resize(image_torch, new_size, interpolation='bilinear')
    end = time.time()
    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)

print(f"RESIZE KORNIA: Average time: {np.median(times)}")


# test kornia resize with backend

times = []

# it's expensive to move the tensor to cuda, so we do it once
# if mwe move every time, the time will be much higher 10x
image_torch_cuda = image_torch.cuda()

for _ in range(num_iters):
    start = time.time()
    with torch.no_grad():
        kornia.geometry.resize(image_torch_cuda, new_size, interpolation='bilinear')
    end = time.time()
    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)

print(f"RESIZE KORNIA CUDA: Average time: {np.median(times)}")
