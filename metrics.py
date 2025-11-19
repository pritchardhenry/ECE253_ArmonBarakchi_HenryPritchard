import numpy as np
import sys


def ComputePSNR(Img1, Img2):
    """
    Compute the Peak Signal to Noise Ratio (PSNR) in decibles(dB).
    """

    if Img1.size != Img2.size:
        print('ERROR: two images should be in same size in computing PSNR.\n')

        sys.exit()

    Img1 = Img1.astype(np.float64)

    Img2 = Img2.astype(np.float64)

    RMSE = np.sqrt(np.sum((Img1 - Img2) ** 2) / Img1.size)

    return 20 * np.log10(255. / RMSE)