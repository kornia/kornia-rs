import time
import cv2
import numpy as np
import kornia_rs

def bench_opencv_vs_kornia(image_size=(1080, 1920), iters=100, warmup=10):
    print(f"Benchmarking SIFT detectAndCompute on {image_size[1]}x{image_size[0]} image")

    # Generate a random uint8 image
    img = np.random.randint(0, 256, (image_size[0], image_size[1]), dtype=np.uint8)

    # 1. OpenCV SIFT
    sift_cv = cv2.SIFT_create()

    # Warmup OpenCV
    for _ in range(warmup):
        _ = sift_cv.detectAndCompute(img, None)

    start_cv = time.perf_counter()
    for _ in range(iters):
        _ = sift_cv.detectAndCompute(img, None)
    cv_time = (time.perf_counter() - start_cv) * 1000 / iters

    print(f"OpenCV SIFT: {cv_time:.2f} ms/frame")

    # 2. Kornia-RS Host SIFT
    sift_kr = kornia_rs.imgproc.Sift()
    img_f32 = img.astype(np.float32).reshape(image_size[0], image_size[1], 1)

    # Warmup Kornia Host
    for _ in range(warmup):
        _ = sift_kr.detect_and_compute(img_f32)

    start_kr = time.perf_counter()
    for _ in range(iters):
        _ = sift_kr.detect_and_compute(img_f32)
    kr_time = (time.perf_counter() - start_kr) * 1000 / iters

    print(f"Kornia-RS Host SIFT: {kr_time:.2f} ms/frame")

    # 3. Kornia-RS CUDA SIFT (Explicit Copies)
    try:
        stream = kornia_rs.cuda.Stream.default()
        sift_cuda = kornia_rs.imgproc.Sift()

        # Warmup Kornia CUDA (Explicit)
        for _ in range(warmup):
            img_cuda = kornia_rs.image.Image.from_numpy(img_f32).to_cuda(stream)
            _ = sift_cuda.detect_and_compute(img_cuda)

        start_cuda = time.perf_counter()
        for _ in range(iters):
            img_cuda = kornia_rs.image.Image.from_numpy(img_f32).to_cuda(stream)
            _ = sift_cuda.detect_and_compute(img_cuda)
        cuda_time = (time.perf_counter() - start_cuda) * 1000 / iters

        print(f"Kornia-RS CUDA (Explicit) SIFT: {cuda_time:.2f} ms/frame")
    except AttributeError:
        print("CUDA not available for Kornia-RS.")

if __name__ == "__main__":
    bench_opencv_vs_kornia((480, 640), iters=50)
    print("---")
    bench_opencv_vs_kornia((720, 1280), iters=20)
    print("---")
    bench_opencv_vs_kornia((1080, 1920), iters=10)
