from bm3d_step1_original import BM3D_Step1
import numpy as np
from bm3d_step1_original import AddNoise
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
class BM3DConfig:
    def __init__(
        self,
        sigma=25,
        BlockSize=8,
        ThreDist=2500,
        MaxMatch=16,
        WindowSize=32,
        spdup_factor=1,
        Kaiser_Window_beta=2.0,
        lamb2d=0.0,
        lamb3d=0.0,
        Transform_Type="DCT",
        Wavelet_Type="db1"
    ):
        self.sigma = sigma
        self.BlockSize = BlockSize
        self.ThreDist = ThreDist
        self.MaxMatch = MaxMatch
        self.WindowSize = WindowSize
        self.spdup_factor = spdup_factor
        self.Kaiser_Window_beta = Kaiser_Window_beta
        self.lamb2d = lamb2d
        self.lamb3d = lamb3d
        self.Transform_Type = Transform_Type
        self.Wavelet_Type = Wavelet_Type


def sample_sig():
    return np.random.uniform(40, 40)

def sample_wave():
    return np.random.choice(['haar', 'bior4.4', 'db2', 'db3', 'db4', 'sym8', 'sym4'])

def sample_lamb3d():
    return np.random.uniform(0,30)

def sample_lamb2d():
    return np.random.uniform(2)
def random_search():
    NUM_SAMPLES = 500  # choose how many random tests to run

    best_results = []
    best_gain = -1e9

    for trial in range(NUM_SAMPLES):

        # -------------------
        # 1. Sample parameters
        # -------------------
        sigma = sample_sig()
        wave = sample_wave()
        lamb3d = sample_lamb3d()
        lamb2d = sample_lamb2d()

        config = BM3DConfig(
            sigma=sigma,
            BlockSize=8,
            ThreDist=2500,
            MaxMatch=32,
            WindowSize=39,
            spdup_factor=3,
            Kaiser_Window_beta=2.5,
            lamb2d=lamb2d,
            lamb3d=lamb3d,
            Transform_Type="DCT_Wavelet",
            Wavelet_Type=wave
        )

        # -------------------
        # 2. Load + prepare image
        # -------------------
        img = plt.imread('istockphoto-1149340384-612x612.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        noisy_img = AddNoise(img, sigma)
        basic_img = BM3D_Step1(noisy_img, config)


        # 5. Compute PSNRs
        # -------------------


        noisy_psnr = psnr(img, noisy_img, data_range=1.0)
        recon_psnr = psnr(img, basic_img, data_range=1.0)

        gain = recon_psnr - noisy_psnr

        noisy_ssim = ssim(img, noisy_img, data_range=1.0)
        recon_ssim = ssim(img, basic_img, data_range=1.0)
        ssim_gain = recon_ssim - noisy_ssim
        # -------------------
        # 6. Update best results
        # -------------------
        if ssim_gain > best_gain:
            best_gain = ssim_gain
            best_results.append({
                "wave": wave,
                "SSIM_gain": ssim_gain,
                "lamb3d": lamb3d,
                "noisy_psnr": noisy_psnr,
                "recon_psnr": recon_psnr,
                "lamb2d": lamb2d,
                "PSNR Gain": gain
            })
            print("🔥 New Best Params:", best_results[-1])

random_search()
