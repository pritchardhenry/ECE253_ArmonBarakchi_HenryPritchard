import numpy as np
from PIL import Image
import math
import scipy.io
import scipy.linalg
from os.path import dirname, join
from skimage.transform import resize  # <-- replacement for scipy.misc.imresize
import scipy.ndimage
import scipy.special
import matplotlib.pyplot as plt

# Precomputed gamma range
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]

    left_mean_sqrt = np.sqrt(np.average(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if len(right_data) > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    imdata2_mean = np.mean(imdata2)
    r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2)) if imdata2_mean != 0 else np.inf

    rhat_norm = r_hat * (((gamma_hat**3 + 1) * (gamma_hat + 1)) / (gamma_hat**2 + 1)**2)

    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)

    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt


def paired_product(new_im):
    shifts = [
        np.roll(new_im, 1, axis=1),
        np.roll(new_im, 1, axis=0),
        np.roll(np.roll(new_im, 1, axis=0), 1, axis=1),
        np.roll(np.roll(new_im, 1, axis=0), -1, axis=1),
    ]
    return tuple(s * new_im for s in shifts)


def gen_gauss_window(lw, sigma):
    lw = int(lw)
    sigma = float(sigma)
    weights = np.zeros(2*lw + 1, dtype=np.float32)
    weights[lw] = 1.0
    sd = sigma * sigma
    sum_w = 1.0
    for i in range(1, lw + 1):
        w = np.exp(-0.5 * i * i / sd)
        weights[lw + i] = weights[lw - i] = w
        sum_w += 2 * w
    return weights / sum_w


def compute_image_mscn_transform(image, C=1, avg_window=None):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)

    assert image.ndim == 2
    image = image.astype(np.float32)

    mu_image = scipy.ndimage.correlate(image, avg_window[:, None], mode='reflect')
    mu_image = scipy.ndimage.correlate(mu_image, avg_window[None, :], mode='reflect')

    var_image = scipy.ndimage.correlate(image**2, avg_window[:, None], mode='reflect')
    var_image = scipy.ndimage.correlate(var_image, avg_window[None, :], mode='reflect')
    var_image = np.sqrt(np.abs(var_image - mu_image**2))

    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    feats = [aggd_features(mscncoefs)[0], (aggd_features(mscncoefs)[2] + aggd_features(mscncoefs)[3]) / 2.0]
    for p in paired_product(mscncoefs):
        feats.extend(aggd_features(p)[:4])
    return np.array(feats)


def extract_patches(img, patch_size):
    h, w = img.shape
    patches = [
        img[j:j+patch_size, i:i+patch_size]
        for j in range(0, h - patch_size + 1, patch_size)
        for i in range(0, w - patch_size + 1, patch_size)
    ]
    return np.array([_niqe_extract_subband_feats(p) for p in patches])


def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)

    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = params["pop_mu"].ravel()
    pop_cov = params["pop_cov"]

    assert inputImgData.shape[0] > 192 and inputImgData.shape[1] > 192

    # Replace scipy.misc.imresize with skimage resize
    img2 = resize(inputImgData, (inputImgData.shape[0] // 2, inputImgData.shape[1] // 2),
                  anti_aliasing=True)

    mscn1, _, _ = compute_image_mscn_transform(inputImgData)
    mscn2, _, _ = compute_image_mscn_transform(img2)

    feats_lvl1 = extract_patches(mscn1, patch_size)
    feats_lvl2 = extract_patches(mscn2, patch_size // 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    return float(np.sqrt(np.dot(np.dot(X, pinvmat), X)))


# Testing code
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Load image as grayscale (L mode)
    ref1 = np.array(Image.open("/Users/armonbarakchi/Desktop/ECE253_ArmonBarakchi_HenryPritchard/dog.png").convert('L')).astype(np.float32) / 255.0

    # Generate distorted version with matching shape
    dis1 = np.clip(ref1 + np.random.normal(0.15, size=ref1.shape), 0, 1)

    print(ref1.shape)  # (424, 612)
    print(dis1.shape)  # (424, 612)

    print('NIQE ref image 1: %.3f' % niqe(ref1))
    print('NIQE distorted image 1: %.3f' % niqe(dis1))
