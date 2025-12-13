import numpy as np
import cv2
import bm3d
import kagglehub
import os
from tqdm import tqdm
# ----------------------------------------------------
# Motion PSF
# ----------------------------------------------------
def motion_psf(length=12, angle_deg=8):
    psf = np.zeros((length, length), np.float32)
    c = length // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)

    for i in range(length):
        x = int(c + (i - c) * dx)
        y = int(c + (i - c) * dy)
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1.0
    return psf / psf.sum()

def conv(img, k):
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)

def convT(img, k):
    return cv2.filter2D(img, -1, np.flip(k, (0,1)), borderType=cv2.BORDER_REPLICATE)


def pnp_bm3d_deblur(y, psf, iters=25, tau=0.25, sigma_denoise=5/255):
    x = y.copy()
    for k in tqdm(range(iters)):
        Ax = conv(x, psf)
        grad = convT(Ax - y, psf)
        # print("Grad")

        x_half = x - tau * grad
        x_half = np.clip(x_half, 0, 1)

        x = bm3d.bm3d(x_half, sigma_psd=sigma_denoise)
        # print("BM3D")
        x = np.clip(x, 0, 1)

    return x


if __name__ == "__main__":
    root = kagglehub.dataset_download("harshraone/urban100")
    TEST_IMAGES_DIRECTORY = os.path.join(
        root, "Urban 100", "X2 Urban100", "X2", "HIGH X2 Urban"
    )
    TEST_IMAGES_DIRECTORY = "50photosBlurry"

    fname = "IMG_5568.JPG"
    original = cv2.imread(os.path.join(TEST_IMAGES_DIRECTORY, fname)).astype(np.float32) / 255.0

    psf = motion_psf(length=20, angle_deg=90)

    blurred = original

    noise_sigma = 3/255.0
    noisy = blurred
    noisy = np.clip(noisy, 0, 1)
    cv2.imwrite("blurred.png", (noisy*255).astype(np.uint8))

    print("Running PnP-BM3D...")
    deblurred = pnp_bm3d_deblur(
        y=noisy,
        psf=psf,
        iters=20,
        tau=.2,
        sigma_denoise=noise_sigma
    )

    cv2.imwrite("original.png", (original*255).astype(np.uint8))
    cv2.imwrite("pnp_bm3d_deblurred.png", (deblurred*255).astype(np.uint8))

    print("Saved results.")
