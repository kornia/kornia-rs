#!/usr/bin/env python3
"""OpenCV Integral & Laplacian bench"""
import time
import numpy as np
import cv2

SIZES = [(1920, 1080), (3840, 2160)]
REPS = 100
WARMUP = 30

def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]

def run_laplacian(label, w, h, data):
    for _ in range(WARMUP):
        cv2.Laplacian(data, cv2.CV_16S, ksize=3)
    samples = []
    for _ in range(REPS):
        t = time.perf_counter()
        cv2.Laplacian(data, cv2.CV_16S, ksize=3)
        samples.append(time.perf_counter() - t)

    mn, md = min(samples), median(samples)
    mu = sum(samples) / len(samples)
    pix_per_s = (w * h) / md / 1e6
    print(f"opencv,{label},{w}x{h},{mn*1e6:.1f},{md*1e6:.1f},{mu*1e6:.1f},{pix_per_s:.1f}")

def run_integral(label, w, h, data):
    for _ in range(WARMUP):
        cv2.integral(data, sdepth=cv2.CV_32F)
    samples = []
    for _ in range(REPS):
        t = time.perf_counter()
        cv2.integral(data, sdepth=cv2.CV_32F)
        samples.append(time.perf_counter() - t)

    mn, md = min(samples), median(samples)
    mu = sum(samples) / len(samples)
    pix_per_s = (w * h) / md / 1e6
    print(f"opencv,{label},{w}x{h},{mn*1e6:.1f},{md*1e6:.1f},{mu*1e6:.1f},{pix_per_s:.1f}")

def main():
    print("# CSV: impl,op,size,min_us,med_us,mean_us,Mpix_s")
    for w, h in SIZES:
        data = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        run_laplacian("laplacian_u8_3x3", w, h, data)
        run_integral("integral_image_u8", w, h, data)

if __name__ == "__main__":
    main()
