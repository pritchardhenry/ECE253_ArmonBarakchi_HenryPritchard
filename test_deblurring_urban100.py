from gaussian_blur import *
from utils import get_config
from bm3d_step1_original import *
from gaussian_blur import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from GraphingFunctions import *
import cv2
import time
import numpy as np
import sys
import shutil
import os
import kagglehub
import json

if __name__ == "__main__":
    config = get_config()
    blur_sigma = config.blur_sigma
    blur_kernel_size = config.blur_kernel_size

    # File management
    BASE_DIRECTORY = "urban100_deblurring_results"
    if os.path.exists(BASE_DIRECTORY):
        print("Deleting old")
        shutil.rmtree(BASE_DIRECTORY)
    os.makedirs(BASE_DIRECTORY)
    TEST_IMAGES_DIRECTORY = kagglehub.dataset_download("harshraone/urban100")
    TEST_IMAGES_DIRECTORY = os.path.join(TEST_IMAGES_DIRECTORY, "Urban 100", "X2 Urban100", "X2", "HIGH X2 Urban")

    image_files = sorted(f for f in os.listdir(TEST_IMAGES_DIRECTORY) if f.lower().endswith(".png"))

    wavelets = {
        "haar": {"transform": "DCT_Wavelet", "lambda": 6, "results": []},
        "db2": {"transform": "DCT_Wavelet", "lambda": 8, "results": []},
        "bior1.3": {"transform": "DCT_Wavelet", "lambda": 6.2, "results": []},
        "dct": {"transform": "3D_DCT", "lambda": 2.7, "results": []}
    }
    for w in wavelets:
        wavelet_dir = os.path.join(BASE_DIRECTORY, w)
        os.makedirs(wavelet_dir, exist_ok=True)

    blur_op = GaussianBlurOp(blur_sigma, blur_kernel_size)
    for img_file in image_files:

        img_path = os.path.join(TEST_IMAGES_DIRECTORY, img_file)
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_img = blur_op(test_img)
        test_img = AddNoise(original_image, config.sigma)

        for w in wavelets:
            config.lamb3d = wavelets[w]["lambda"]
            config.Wavelet_Type = w
            config.Transform_Type = wavelets[w]["transform"]

            y = test_img
            step_size = config.step_size
            gamma = config.gamma

            for i in range(0, config.num_steps):
                x = y - step_size * gamma * blur_op.blur(blur_op.blur(y) - test_img)
                y = BM3D_Step1(x, config)
            denoised_img = y

            pdf_file = os.path.splitext(img_file)[0] + ".pdf"
            plot_path = os.path.join(BASE_DIRECTORY, w, pdf_file)

            plotGraph([denoised_img], [test_img], [original_image], save_path=plot_path)

            den_psnr = psnr(denoised_img, original_image, data_range=255)
            den_ssim = ssim(denoised_img, original_image, data_range=255)

            wavelets[w]["results"].append({"image": img_file, "psnr": float(den_psnr), "ssim": float(den_ssim)})

    json_path = os.path.join(BASE_DIRECTORY, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(wavelets, f, indent=4)
    print(f"Saved metrics to {json_path}")