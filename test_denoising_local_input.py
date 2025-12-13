import argparse
from utils import get_config
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

    test_image_path = config.test_img_path
    sigma = config.sigma
    original_image = cv2.imread(test_image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    noisy_img = AddNoise(img, sigma)

    basic_img = BM3D_Step1(noisy_img, config)

    basic_PSNR = psnr(img, basic_img, data_range=255)
    basic_ssim = ssim(img, basic_img, data_range=255)

    print('The PSNR of the denoised image is {} dB.\n'.format(basic_PSNR))
    print('The SSIM of the denoised image is {} dB.\n'.format(basic_ssim))
    plotGraph([basic_img], [noisy_img], [img])
