from scipy.fftpack import dct, idct
import numpy as np
import pywt
import pywt
import numpy as np

def dct_dwt_transform(BlockGroup, sigma, lamb3d, wavelet='haar', level=1,  DEBUG=False):
    B, N, M = BlockGroup.shape
    BlockGroup = BlockGroup.astype(np.float64).copy()

    ThreValue = lamb3d * sigma
    nonzero_cnt = 0

    w = pywt.Wavelet(wavelet)


    total_positions = N * M
    zero_frac_counter = 0
    kept_frac_list = []

    for i in range(N):
        for j in range(M):
            vector = BlockGroup[:, i, j]
            coeffs = pywt.wavedec(vector, wavelet=w, level=level, axis=0)
            if DEBUG and i == 0 and j == 0:
                print(f"[DEBUG] requested level={level}, actual levels = {len(coeffs) - 1}")
            new_coeffs = [coeffs[0]]
            for lvl, cD in enumerate(coeffs[1:], start=1):
                abs_cD = np.abs(cD)
                mask = abs_cD >= ThreValue
                kept = int(mask.sum())
                total = cD.size
                frac_kept = kept / total if total > 0 else 0.0
                kept_frac_list.append(frac_kept)
                nonzero_cnt += kept

                if frac_kept == 0:
                    zero_frac_counter += 1

                if DEBUG and (i == 0 and j < 3):
                    print(f"[i={i}, j={j}] level={lvl} "
                          f"max|cD|={abs_cD.max():.2f} kept_frac={frac_kept:.3f}")

                new_coeffs.append(cD * mask)

            reconstructed = pywt.waverec(new_coeffs, wavelet=w, axis=0)
            BlockGroup[:, i, j] = reconstructed[:B]

    return BlockGroup, nonzero_cnt

