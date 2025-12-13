import argparse
from utils import get_config
from gaussian_blur import *
from bm3d_step1_original import BM3D_Step1, AddNoise
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import time
import numpy as np
import sys
import kagglehub
import os
import matplotlib.pyplot as plt
from GraphingFunctions import *


if __name__ == "__main__":
    config = get_config()
    blur_sigma = config.blur_sigma
    blur_kernel_size = config.blur_kernel_size
    step_size = config.step_size
    gamma = config.gamma
    blur_op = GaussianBlurOp(blur_sigma, blur_kernel_size)

    test_image_path = config.test_img_path
    sigma = config.sigma
    original_image = cv2.imread(test_image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img = blur_op(img)
    img = AddNoise(original_image, config.sigma)
    y= img
    for i in range(0, config.num_steps):
        x = y - step_size * gamma * blur_op.blur(blur_op.blur(y) - img)
        y = BM3D_Step1(x, config)
    basic_img = y

    basic_PSNR = psnr(img, basic_img, data_range=255)
    basic_ssim = ssim(img, basic_img, data_range=255)

    print('The PSNR of the denoised image is {} dB.\n'.format(basic_PSNR))
    print('The SSIM of the denoised image is {} dB.\n'.format(basic_ssim))
    plotGraph([basic_img], [img], [original_image])
