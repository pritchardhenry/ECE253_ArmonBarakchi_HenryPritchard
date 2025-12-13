import argparse
import numpy as np



def get_config():
    parser = argparse.ArgumentParser(description='config')

    # File Management
    parser.add_argument("--first_pass_mode", type=str, default="original")
    parser.add_argument("--test_img_path", type=str, default="istockphoto-1149340384-612x612.jpg")
    parser.add_argument("--output_path", type=str, default="denoised.png")

    # Noise & filtering parameters
    parser.add_argument("--sigma", type=float, default=25, help="Noise standard deviation (default: 25)")
    parser.add_argument("--lamb2d", type=float, default=2.0, help="2D transform threshold scaling (default: 2.0)")
    parser.add_argument("--lamb3d", type=float, default=2.7, help="3D transform threshold scaling (default: 2.7)")
    parser.add_argument("--level", type=int, default=1, help="level of DWT (default: 1). 3 level used in testing.")
    parser.add_argument("--Transform_Type", type=str, default="3D_DCT", help="3D Transform type (default = 3D_DCT)\nOptions:\n DCT_Wavelet, 3D_DCT. ")
    parser.add_argument("--Wavelet_Type", type=str, default="haar", help="Wavelet type for 3D transform (default: haar). See https://pywavelets.readthedocs.io/en/latest/ for details on the available options.")

    # Grouping parameters
    parser.add_argument("--ThreDist", type=float, default=2500, help="Threshold distance for block matching (default: 2500)")
    parser.add_argument("--MaxMatch", type=int, default=16, help="Max number of similar blocks to group (default: 16)")
    parser.add_argument("--BlockSize", type=int, default=8, help="Block size (default: 8)")
    parser.add_argument("--WindowSize", type=int, default=39, help="Search window size (default: 39)")
    parser.add_argument("--Kaiser_Window_beta", type=float, default=2.0, help="Beta parameter for Kaiser window (default: 2.0)")
    parser.add_argument("--spdup_factor", type=int, default=3, help="Pixel jump step for reference blocks (default: 3)")

    args = parser.parse_args()
    return args



def AddNoise(Img, sigma):
    GuassNoise = np.random.normal(0, sigma, Img.shape)
    noisyImg = Img + GuassNoise
    return noisyImg