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

BASE_DIRECTORY = "urban100_results"
TEST_IMAGES_DIRECTORY = kagglehub.dataset_download("harshraone/urban100")
TEST_IMAGES_DIRECTORY = os.path.join(TEST_IMAGES_DIRECTORY, "Urban 100", "X2 Urban100", "X2", "LOW X2 Urban")




if __name__ == "__main__":
    config = get_config()

    test_image_path = config.test_img_path
    output_path = config.output_path
    sigma = config.sigma
    original_image = cv2.imread(os.path.join(TEST_IMAGES_DIRECTORY, "img_087_SRF_2_LR.png"))
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    noisy_img = AddNoise(img, sigma)

    # noisy_psnr = psnr(noisy_img, img, data_range=255)
    # noisy_ssim = ssim(noisy_img, img, data_range=255)
    # print('The PSNR of the noisy image is {} dB.\n'.format(noisy_psnr))
    # print('The SSIM of the noisy image is {} dB.\n'.format(noisy_ssim))

    # print('The PSNR of noisy image is {} dB.\n'.format(starting_psnr))
    # print('The SSIM of ssim image is {} dB.\n'.format(starting_ssim))


    basic_img = BM3D_Step1(noisy_img, config)

    basic_PSNR = psnr(img, basic_img, data_range=255)
    basic_ssim = ssim(img, basic_img, data_range=255)

    print('The PSNR of the denoised image is {} dB.\n'.format(basic_PSNR))
    print('The SSIM of the denoised image is {} dB.\n'.format(basic_ssim))
    plt.imshow(basic_img, cmap='gray')
    plt.show()

    # basic_img_uint = np.zeros(img.shape)
    #
    # cv2.normalize(basic_img, basic_img_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    #
    # basic_img_uint = basic_img_uint.astype(np.uint8)
    #
    #
    #
    # if cv2.imwrite(output_path, basic_img_uint) == True:
    #
    #     print('Basic estimate has been saved successfully.\n')
    #
    #     step1_time = time.time()
    #
    #     print('The running time of basic estimate is', step1_time - start_time, 'seconds.\n')
    #
    # else:
    #
    #     print('ERROR: basic estimate is not reconstructed successfully.\n')
    #
    #     sys.exit()