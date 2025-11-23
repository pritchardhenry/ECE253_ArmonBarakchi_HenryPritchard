from scipy.fftpack import dct, idct
import numpy as np
import pywt


def dct_dwt_transform(BlockGroup, lamb3d, sigma, wavelet='haar'):
    B, N, M = BlockGroup.shape
    BlockGroup = np.asarray(BlockGroup).copy()

    ThreValue = lamb3d * sigma
    nonzero_cnt = 0

    for b in range(B):
        BlockGroup[b] = dct(dct(BlockGroup[b], axis=0, norm='ortho'), axis=1, norm='ortho')
    # if wavelet == 'haar':
    for i in range(N):
        for j in range(M):
            vector = BlockGroup[:, i, j]
            coeffs = pywt.wavedec(vector, wavelet=wavelet, axis=0)
            thresholded_coeffs = coeffs.copy()
            for k in range(1, len(coeffs)):
                mask = np.abs(coeffs[k]) >= ThreValue
                thresholded_coeffs[k] = coeffs[k] * mask
                nonzero_cnt += np.count_nonzero(mask)
            reconstructed = pywt.waverec(coeffs, wavelet=wavelet, axis=0)
            BlockGroup[:, i, j] = reconstructed[:B]
    # if wavelet.startswith('db'):
    #     for i in range(N):
    #         for j in range(M):
    #             vector = BlockGroup[:, i, j]
    #             coeffs = pywt.wavedec(vector, wavelet=wavelet, axis=0)
    #             thresholded_coeffs = coeffs.copy()
    #             for k in range(1, len(coeffs)):
    #                 mask = np.abs(coeffs[k]) >= ThreValue
    #                 thresholded_coeffs[k] = coeffs[k] * mask
    #                 nonzero_cnt += np.count_nonzero(mask)
    #
    #             reconstructed = pywt.waverec(thresholded_coeffs, wavelet=wavelet, axis=0)
    #             BlockGroup[:, i, j] = reconstructed[:B]  # truncate if needed

    for b in range(B):
        BlockGroup[b] = idct(idct(BlockGroup[b], axis=1, norm='ortho'), axis=0, norm='ortho')

    return BlockGroup, nonzero_cnt

def full_wavelet_3d_transform(BlockGroup, lamb3d, sigma, wavelet='haar'):
    """
    Applies Haar or Daubechies transform along all 3 axes (0,1,2), thresholds, then reconstructs.

    Parameters:
    -----------
    BlockGroup : ndarray of shape (B, N, M)
    lamb3d : float
    sigma : float
    wavelet : str
        Should be 'haar' or any supported wavelet.

    Returns:
    --------
    BlockGroup : ndarray (same shape), reconstructed from thresholded coefficients
    nonzero_cnt : int, total number of nonzero wavelet coefficients after thresholding
    """
    BlockGroup = np.asarray(BlockGroup).copy()
    ThreValue = lamb3d * sigma
    nonzero_cnt = 0

    # Step 1: Apply wavelet along axis=0
    for i in range(BlockGroup.shape[1]):
        for j in range(BlockGroup.shape[2]):
            vec = BlockGroup[:, i, j]
            coeffs = pywt.wavedec(vec, wavelet=wavelet, axis=0)
            thresholded = coeffs.copy()
            for k in range(1, len(coeffs)):
                mask = np.abs(coeffs[k]) >= ThreValue
                thresholded[k] = coeffs[k] * mask
                nonzero_cnt += np.count_nonzero(mask)
            BlockGroup[:, i, j] = pywt.waverec(thresholded, wavelet=wavelet, axis=0)[:BlockGroup.shape[0]]

    # Step 2: Apply wavelet along axis=1
    for b in range(BlockGroup.shape[0]):
        for j in range(BlockGroup.shape[2]):
            vec = BlockGroup[b, :, j]
            coeffs = pywt.wavedec(vec, wavelet=wavelet, axis=0)
            thresholded = coeffs.copy()
            for k in range(1, len(coeffs)):
                mask = np.abs(coeffs[k]) >= ThreValue
                thresholded[k] = coeffs[k] * mask
                nonzero_cnt += np.count_nonzero(mask)
            BlockGroup[b, :, j] = pywt.waverec(thresholded, wavelet=wavelet, axis=0)[:BlockGroup.shape[1]]

    # Step 3: Apply wavelet along axis=2
    for b in range(BlockGroup.shape[0]):
        for i in range(BlockGroup.shape[1]):
            vec = BlockGroup[b, i, :]
            coeffs = pywt.wavedec(vec, wavelet=wavelet, axis=0)
            thresholded = coeffs.copy()
            for k in range(1, len(coeffs)):
                mask = np.abs(coeffs[k]) >= ThreValue
                thresholded[k] = coeffs[k] * mask
                nonzero_cnt += np.count_nonzero(mask)
            BlockGroup[b, i, :] = pywt.waverec(thresholded, wavelet=wavelet, axis=0)[:BlockGroup.shape[2]]

    return BlockGroup, nonzero_cnt

