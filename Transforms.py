from scipy.fftpack import dct, idct
import numpy as np
import pywt
import pywt
import numpy as np

def dct_dwt_transform(BlockGroup, sigma, lamb3d, wavelet='haar', DEBUG=True):
    """
    BlockGroup: shape (B, N, M), *already* in 2D DCT domain.
    We apply 1D wavelet along the group dimension (B) and threshold detail coeffs.
    """
    B, N, M = BlockGroup.shape
    BlockGroup = BlockGroup.astype(np.float64).copy()

    ThreValue = lamb3d * sigma
    nonzero_cnt = 0

    # if DEBUG:
    #     print("="*60)
    #     print(f"[DEBUG] Group Size B = {B}")
    #     print(f"[DEBUG] Threshold ThreValue = {ThreValue:.4f}")
    #     if B < 4:
    #         print("⚠️  WARNING: Group size B is very small for 3D DWT.")

    # pick a level that is valid, but at least 1
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(B, w.dec_len)
    L = min(3, max(1, max_level))   # 1 ≤ L ≤ 3

    total_positions = N * M
    zero_frac_counter = 0
    kept_frac_list = []

    for i in range(N):
        for j in range(M):
            vector = BlockGroup[:, i, j]  # (B,)

            # multi-level 1D DWT along group dimension
            coeffs = pywt.wavedec(vector, wavelet=w, level=L, axis=0)
            if DEBUG and i == 0 and j == 0:
                print(f"[DEBUG] requested level L={L}, actual levels = {len(coeffs) - 1}")

            # coeffs[0] = approximation, coeffs[1:] = detail levels
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

            # inverse DWT
            reconstructed = pywt.waverec(new_coeffs, wavelet=w, axis=0)
            BlockGroup[:, i, j] = reconstructed[:B]

    # NOTE: no 2D inverse DCT here! Aggregation will do idct2D as before.
    #
    # if DEBUG:
    #     print("-"*60)
    #     mean_kept = np.mean(kept_frac_list) if kept_frac_list else 0.0
    #     print(f"[SUMMARY] B = {B}")
    #     print(f"[SUMMARY] ThreValue = {ThreValue:.2f}")
    #     print(f"[SUMMARY] Average frac_kept = {mean_kept:.3f}")
    #     print(f"[SUMMARY] Zero-kept positions = {zero_frac_counter}/{total_positions}")
    #     if mean_kept < 0.05:
    #         print("❗ WARNING: shrinkage too aggressive — λ₃D too high.")
    #     if mean_kept > 0.7:
    #         print("❗ WARNING: shrinkage too weak — λ₃D too low.")
    #     print("="*60)

    return BlockGroup, nonzero_cnt

