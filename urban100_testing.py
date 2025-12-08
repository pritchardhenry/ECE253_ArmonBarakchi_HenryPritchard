from utils import get_config
from bm3d_step1_original import BM3D_Step1, AddNoise
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import time
import numpy as np
import sys
import shutil
import os
import kagglehub
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = get_config()


    # File management
    BASE_DIRECTORY = "urban100_results"
    if os.path.exists(BASE_DIRECTORY):
        print("Deleting old")
        shutil.rmtree(BASE_DIRECTORY)
    os.makedirs(BASE_DIRECTORY)
    TEST_IMAGES_DIRECTORY = kagglehub.dataset_download("harshraone/urban100")
    TEST_IMAGES_DIRECTORY = os.path.join(TEST_IMAGES_DIRECTORY, "Urban 100", "X2 Urban100", "X2", "HIGH X2 Urban")

    original_image = cv2.imread(os.path.join(TEST_IMAGES_DIRECTORY, "img_087_SRF_2_HR.png"))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    test_img = AddNoise(original_image, config.sigma)
    noisy_psnr = psnr(test_img, original_image, data_range=255)
    noisy_ssim = ssim(test_img, original_image, data_range=255)
    print('The PSNR of the noisy image is {} dB.\n'.format(noisy_psnr))
    print('The SSIM of the noisy image is {} dB.\n'.format(noisy_ssim))

    for lamb3d in range(0, 200, 8):
        config.lamb3d = lamb3d
        print(f"Testing Lambda 3D of {lamb3d}")



        denoised_img = BM3D_Step1(test_img, config)
        denoised_psnr = psnr(denoised_img, original_image, data_range=255)
        denoised_ssim = ssim(denoised_img, original_image, data_range=255)
        print('The PSNR of the denoised image is {} dB.\n'.format(denoised_psnr))
        print('The SSIM of the denoised image is {} dB.\n'.format(denoised_ssim))

        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.imshow(original_image, cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(test_img.clip(0,255), cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(denoised_img.clip(0,255), cmap='gray')

        plt.savefig(os.path.join(BASE_DIRECTORY, f"lamb3d_of_{lamb3d}.png"))
        # plt.show()
