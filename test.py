import argparse
from utils import get_config
from bm3d_step1_original import BM3D_Step1, AddNoise
from metrics import ComputePSNR
import cv2
import time
import numpy as np
import sys





if __name__ == "__main__":
    config = get_config()

    test_image_path = config.test_img_path
    sigma = config.sigma
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    noisy_img = AddNoise(img, sigma)

    starting_psnr = ComputePSNR(img, noisy_img)

    print('The PSNR of noisy image is {} dB.\n'.format(starting_psnr))

    cv2.imwrite('noisy.png', noisy_img)

    start_time = time.time()

    basic_img = BM3D_Step1(noisy_img, config)

    basic_PSNR = ComputePSNR(img, basic_img)

    print('The PSNR of basic image is {} dB.\n'.format(basic_PSNR))

    basic_img_uint = np.zeros(img.shape)

    cv2.normalize(basic_img, basic_img_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    basic_img_uint = basic_img_uint.astype(np.uint8)

    if cv2.imwrite('denoise_original.png', basic_img_uint) == True:

        print('Basic estimate has been saved successfully.\n')

        step1_time = time.time()

        print('The running time of basic estimate is', step1_time - start_time, 'seconds.\n')

    else:

        print('ERROR: basic estimate is not reconstructed successfully.\n')

        sys.exit()