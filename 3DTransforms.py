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
    if wavelet == 'haar':
        for i in range(N):
            for j in range(M):
                vector = BlockGroup[:, i, j]
                coeffs = pywt.wavedec(vector, wavelet=wavelet, axis=0)
                vector[abs(vector[:]) < ThreValue] = 0.
                nonzero_cnt += np.nonzero(vector)[0].size
                reconstructed = pywt.waverec(coeffs, wavelet=wavelet, axis=0)
                BlockGroup[:, i, j] = reconstructed[:B]
    if wavelet.startswith('db'):
        for i in range(N):
            for j in range(M):
                vector = BlockGroup[:, i, j]
                coeffs = pywt.wavedec(vector, wavelet=wavelet, axis=0)
                thresholded_coeffs = coeffs.copy()
                for k in range(1, len(coeffs)):
                    mask = np.abs(coeffs[k]) >= ThreValue
                    thresholded_coeffs[k] = coeffs[k] * mask
                    nonzero_cnt += np.count_nonzero(mask)

                reconstructed = pywt.waverec(thresholded_coeffs, wavelet=wavelet, axis=0)
                BlockGroup[:, i, j] = reconstructed[:B]  # truncate if needed

    for b in range(B):
        BlockGroup[b] = idct(idct(BlockGroup[b], axis=1, norm='ortho'), axis=0, norm='ortho')

    return BlockGroup, nonzero_cnt

def wavelet_transforms(BlockGroup, wavelet_order=2):