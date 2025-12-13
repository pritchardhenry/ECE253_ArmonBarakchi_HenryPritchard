import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel2d(size=5, sigma=2.0, device="cpu"):
    ax = torch.arange(size, device=device) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


class GaussianBlurOp:
    def __init__(self, size=7, sigma=2.0, device="cpu"):
        self.kernel = gaussian_kernel2d(size, sigma, device)
        self.kernel = self.kernel[None, None, :, :]
        self.size = size
        self.device = device

    def blur(self, img):

        if isinstance(img, np.ndarray):
            img_t = torch.tensor(img, dtype=torch.float32, device=self.device)
        else:
            img_t = img.to(self.device).float()

        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
            channels = 1
        elif img_t.ndim == 3 and img_t.shape[-1] in [1,3]:
            img_t = img_t.permute(2,0,1).unsqueeze(0)
            channels = img_t.shape[1]
        elif img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
            channels = img_t.shape[1]
        else:
            raise ValueError("Image must be 2D or 3D")

        k = self.kernel.repeat(channels, 1, 1, 1)


        pad = self.size // 2
        img_p = F.pad(img_t, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(img_p, k, groups=channels)

        out = out.squeeze(0)
        if out.shape[0] == 1:
            out = out[0]
        else:
            out = out.permute(1,2,0)

        return out.cpu().numpy()


import torch
import torch.nn.functional as F
import numpy as np


def motion_kernel_2d(length=15, angle=0.0, device="cpu"):

    kernel = torch.zeros((length, length), dtype=torch.float32, device=device)
    center = length // 2

    theta = np.deg2rad(angle)
    dx = np.cos(theta)
    dy = np.sin(theta)

    for i in range(length):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0

    kernel = kernel / kernel.sum()
    return kernel


class MotionBlurOp:

    def __init__(self, length=15, angle=0.0, device="cpu"):
        k = motion_kernel_2d(length, angle, device)
        self.kernel = k[None, None, :, :]    # 1×1×H×W
        self.length = length
        self.device = device

    def blur(self, img):

        if isinstance(img, np.ndarray):
            img_t = torch.tensor(img, dtype=torch.float32, device=self.device)
        else:
            img_t = img.to(self.device).float()


        if img_t.ndim == 2:                       # H W
            img_t = img_t.unsqueeze(0).unsqueeze(0)
            channels = 1
        elif img_t.ndim == 3 and img_t.shape[-1] in [1,3]:  # H W C
            img_t = img_t.permute(2,0,1).unsqueeze(0)
            channels = img_t.shape[1]
        elif img_t.ndim == 3:                     # C H W
            img_t = img_t.unsqueeze(0)
            channels = img_t.shape[1]
        else:
            raise ValueError("Unexpected image shape for motion blur.")


        k = self.kernel.repeat(channels, 1, 1, 1)   # C×1×H×W

        pad = self.length // 2
        img_p = F.pad(img_t, (pad, pad, pad, pad), mode="reflect")

        out = F.conv2d(img_p, k, groups=channels)

        out = out.squeeze(0)
        if out.shape[0] == 1:
            out = out[0]
        else:
            out = out.permute(1, 2, 0)

        return out.cpu().numpy()
