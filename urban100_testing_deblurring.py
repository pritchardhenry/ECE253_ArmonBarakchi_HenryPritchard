from gaussian_blur import *
BLUR_SIGMA = 80
BLUR_KERNEL_SIZE = 3
from utils import get_config
from bm3d_step1_original import *
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

if __name__ == "__main__":
    config = get_config()
    config.sigma=40
    blur_op = GaussianBlurOp(size=BLUR_KERNEL_SIZE, sigma = BLUR_SIGMA)


    # File management
    TEST_IMAGES_DIRECTORY = kagglehub.dataset_download("harshraone/urban100")
    TEST_IMAGES_DIRECTORY = os.path.join(TEST_IMAGES_DIRECTORY, "Urban 100", "X2 Urban100", "X2", "HIGH X2 Urban")

    original_image = cv2.imread(os.path.join(TEST_IMAGES_DIRECTORY, "img_087_SRF_2_HR.png"))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    test_img = AddNoise(blur_op.blur(original_image), sigma = 0)
    y = test_img
    step_size = 1
    gamma = 1

    for i in range(0,20):
        x = y-step_size*gamma*blur_op.blur(blur_op.blur(y)-test_img)
        y = BM3D_Step1(x, config)

    # denoised_img = BM3D_Step1(test_img, config)
    # denoised_psnr = psnr(denoised_img, original_image, data_range=1)
    # denoised_ssim = ssim(denoised_img, original_image, data_range=1)
    # print('The PSNR of the denoised image is {} dB.\n'.format(denoised_psnr))
    # print('The SSIM of the denoised image is {} dB.\n'.format(denoised_ssim))

    plotGraph([y], [test_img], [original_image])

