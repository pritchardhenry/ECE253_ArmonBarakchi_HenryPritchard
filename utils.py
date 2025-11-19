import argparse
import numpy as np



def get_config():
    parser = argparse.ArgumentParser(description='config')

    parser.add_argument("--first_pass_mode", type=str, default="original")
    parser.add_argument("--test_img_path", type=str, default="istockphoto-1149340384-612x612.jpg")
    # Noise & filtering parameters
    parser.add_argument("--sigma", type=float, default=25,
                        help="Noise standard deviation (default: 25)")
    parser.add_argument("--lamb2d", type=float, default=2.0,
                        help="2D transform threshold scaling (default: 2.0)")
    parser.add_argument("--lamb3d", type=float, default=2.7,
                        help="3D transform threshold scaling (default: 2.7)")

    # Step 1 parameters
    parser.add_argument("--Step1_ThreDist", type=float, default=2500,
                        help="Threshold distance for block matching (default: 2500)")
    parser.add_argument("--Step1_MaxMatch", type=int, default=16,
                        help="Max number of similar blocks to group (default: 16)")
    parser.add_argument("--Step1_BlockSize", type=int, default=8,
                        help="Block size (default: 8)")
    parser.add_argument("--Step1_spdup_factor", type=int, default=3,
                        help="Pixel jump step for reference blocks (default: 3)")
    parser.add_argument("--Step1_WindowSize", type=int, default=39,
                        help="Search window size (default: 39)")

    # Kaiser window parameter
    parser.add_argument("--Kaiser_Window_beta", type=float, default=2.0,
                        help="Beta parameter for Kaiser window (default: 2.0)")

    args = parser.parse_args()
    return args



def AddNoise(Img, sigma):
    """
    Add Gaussian nosie to an image

    Return:
        nosiy image
    """

    GuassNoise = np.random.normal(0, sigma, Img.shape)

    noisyImg = Img + GuassNoise  # float type noisy image

    #    cv2.normalize(noisyImg, noisyImg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    #
    #    noisyImg = noisyImg.astype(np.uint8)
    #
    #    cv2.imwrite('noisydog.png', noisyImg)
    #
    #    if cv2.imwrite('noisydog.png', noisyImg) == True:
    #
    #        print('Noise has been added to the original image.\n')
    #
    #        return noisyImg
    #
    #    else:
    #
    #        print('Error: adding noise failed.\n')
    #
    #        exit()

    return noisyImg