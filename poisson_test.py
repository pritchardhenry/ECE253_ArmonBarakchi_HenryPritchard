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
    TEST_IMAGES_DIRECTORY = kagglehub.dataset_download("harshraone/urban100")
    TEST_IMAGES_DIRECTORY = os.path.join(TEST_IMAGES_DIRECTORY, "Urban 100", "X2 Urban100", "X2", "LOW X2 Urban")

    original_image = cv2.imread(os.path.join(TEST_IMAGES_DIRECTORY, "img_087_SRF_2_LR.png"))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    original_image = original_image

    PEAK_PHOTONS = 100
    lam = PEAK_PHOTONS * original_image
    y_poisson = np.random.poisson(lam/255)*255
    test_img  = y_poisson / PEAK_PHOTONS

    denoised_img = BM3D_Step1(test_img, config)
    denoised_psnr = psnr(denoised_img, original_image, data_range=1)
    denoised_ssim = ssim(denoised_img, original_image, data_range=1)
    print('The PSNR of the denoised image is {} dB.\n'.format(denoised_psnr))
    print('The SSIM of the denoised image is {} dB.\n'.format(denoised_ssim))

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(test_img, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_img, cmap='gray')
    plt.show()
