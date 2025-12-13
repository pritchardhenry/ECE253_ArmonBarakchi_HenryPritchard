import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import cv2


lpips_loss = lpips.LPIPS(net='alex')

def compute_lpips(img1, img2):
        img1 = np.repeat(img1[..., np.newaxis], 3, axis=2)
        img2 = np.repeat(img2[..., np.newaxis], 3, axis=2)

    t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
    t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()

    return lpips_loss(t1, t2).item()

def plotGraph(clean_imgs, noisy_imgs, original_imgs=None, save_path=None):
    assert len(clean_imgs) == len(noisy_imgs)
    if original_imgs is not None:
        assert len(clean_imgs) == len(original_imgs)

    num_images = len(clean_imgs)
    cols = 3 if original_imgs is not None else 2

    fig = plt.figure(figsize=(3 * cols, 3 * num_images))
    gs = gridspec.GridSpec(num_images, cols, figure=fig,
                           wspace=0.02, hspace=0.15)

    for i in range(num_images):
        col = 0

        if original_imgs is not None:
            ax = fig.add_subplot(gs[i, col])
            ax.imshow(original_imgs[i], cmap='gray')
            ax.set_title("Original", fontsize=12)
            ax.axis('off')
            col += 1

        ax = fig.add_subplot(gs[i, col])
        ax.imshow(noisy_imgs[i], cmap='gray')
        ax.set_title("Noisy", fontsize=12)
        ax.axis('off')

        if original_imgs is not None:
            ps = psnr(original_imgs[i], noisy_imgs[i], data_range=noisy_imgs[i].max()-noisy_imgs[i].min())
            ss = ssim(original_imgs[i], noisy_imgs[i], data_range=noisy_imgs[i].max()-noisy_imgs[i].min(),
                      channel_axis=-1 if noisy_imgs[i].ndim == 3 else None)
            lp = compute_lpips(original_imgs[i], noisy_imgs[i])
            ax.text(0.03, 0.05, f"PSNR: {ps:.2f}\nSSIM: {ss:.3f}\nLPIPS: {lp:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)

        col += 1

        ax = fig.add_subplot(gs[i, col])
        ax.imshow(clean_imgs[i], cmap='gray')
        ax.set_title("Denoised", fontsize=12)
        ax.axis('off')

        if original_imgs is not None:
            ps = psnr(original_imgs[i], clean_imgs[i], data_range=clean_imgs[i].max()-clean_imgs[i].min())
            ss = ssim(original_imgs[i], clean_imgs[i], data_range=clean_imgs[i].max()-clean_imgs[i].min(),
                      channel_axis=-1 if clean_imgs[i].ndim == 3 else None)
            lp = compute_lpips(original_imgs[i], clean_imgs[i])
            ax.text(0.03, 0.05, f"PSNR: {ps:.2f}\nSSIM: {ss:.3f}\nLPIPS: {lp:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as PDF at: {save_path}")
    if save_path is None:
        plt.show()

