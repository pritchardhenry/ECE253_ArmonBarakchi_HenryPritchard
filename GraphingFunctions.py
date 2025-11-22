import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
from niqe import niqe
import cv2


# Assume niqe is already imported from your NIQE implementation
# from your_niqe_module import niqe

lpips_loss = lpips.LPIPS(net='alex')  # Load LPIPS model once

def compute_lpips(img1, img2):
    """Convert numpy image to torch tensor and compute LPIPS score."""
    if img1.ndim == 2:  # grayscale → RGB
        img1 = np.repeat(img1[..., np.newaxis], 3, axis=2)
        img2 = np.repeat(img2[..., np.newaxis], 3, axis=2)

    t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
    t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()

    return lpips_loss(t1, t2).item()


def plotGraph(clean_imgs, noisy_imgs, original_imgs=None, save_path=None):
    """Plot graph of noisy, clean, and optionally original images. Calculates statistics
    and displays them in the plot. To use, need to ensure that the indexes of the img arrays
    line up. i.e: clean[1] should be the cleaned version of noisy[1] and original[1]."""
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

        # ----- Original Image -----
        if original_imgs is not None:
            ax = fig.add_subplot(gs[i, col])
            ax.imshow(original_imgs[i], cmap='gray')
            ax.set_title("Original", fontsize=12)
            ax.axis('off')
            col += 1

        # ----- Noisy Image -----
        ax = fig.add_subplot(gs[i, col])
        ax.imshow(noisy_imgs[i], cmap='gray')
        ax.set_title("Noisy", fontsize=12)
        ax.axis('off')

        if original_imgs is not None:
            ps = psnr(original_imgs[i], noisy_imgs[i], data_range=1)
            ss = ssim(original_imgs[i], noisy_imgs[i], data_range=1,
                      channel_axis=-1 if noisy_imgs[i].ndim == 3 else None)
            lp = compute_lpips(original_imgs[i], noisy_imgs[i])
            ax.text(0.03, 0.05, f"PSNR: {ps:.2f}\nSSIM: {ss:.3f}\nLPIPS: {lp:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)
        else:
            # --- Compute NIQE when original not available ---
            niqe_noisy = niqe(noisy_imgs[i])
            ax.text(0.03, 0.05, f"NIQE: {niqe_noisy:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)

        col += 1

        # ----- Clean Image -----
        ax = fig.add_subplot(gs[i, col])
        ax.imshow(clean_imgs[i], cmap='gray')
        ax.set_title("Clean", fontsize=12)
        ax.axis('off')

        if original_imgs is not None:
            ps = psnr(original_imgs[i], clean_imgs[i], data_range=1)
            ss = ssim(original_imgs[i], clean_imgs[i], data_range=1,
                      channel_axis=-1 if clean_imgs[i].ndim == 3 else None)
            lp = compute_lpips(original_imgs[i], clean_imgs[i])
            ax.text(0.03, 0.05, f"PSNR: {ps:.2f}\nSSIM: {ss:.3f}\nLPIPS: {lp:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)
        else:
            # --- Compute NIQE when original not available ---
            niqe_clean = niqe(clean_imgs[i])
            ax.text(0.03, 0.05, f"NIQE: {niqe_clean:.3f}",
                    fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
                    transform=ax.transAxes)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as PDF at: {save_path}")
    if save_path is None:
        plt.show()


def zoom(imgpath, x_coord, y_coord, scale):
    img = cv2.imread(imgpath)

    # Defining co-ordinates to create a window around the point of interest
    # The window is proportional to size of original image, to maintain aspect ratio
    # Defining the top left, and bottom right co-ordinates
    tl_x = int(x_coord - (float(1) / float(scale) * float(1) / float(2) * img.shape[1]))
    tl_y = int(y_coord - (float(1) / float(scale) * float(1) / float(2) * img.shape[0]))

    br_x = int(tl_x + (float(1) / float(scale) * img.shape[1]))
    br_y = int(tl_y + (float(1) / float(scale) * img.shape[0]))

    # The window is adjusted, if it happens to be going out of the scope of the original image size
    if (tl_x < 0):
        br_x = br_x - tl_x
        tl_x = 0
    if (tl_y < 0):
        br_y = br_y - tl_y
        tl_y = 0

    if (br_x > img.shape[1]):
        tl_x = tl_x - (br_x - img.shape[1])
        br_x = img.shape[1]
    if (br_y > img.shape[0]):
        tl_y = tl_y - (br_y - img.shape[0])
        br_y = img.shape[0]

    # The window is applied over the image to get only the required image portion
    # roi - region of interest
    roi = img[tl_y:br_y, tl_x:br_x, :]

    # The width and height of this roi is stored
    width1 = (roi.shape[1]) - 1
    height1 = (roi.shape[0]) - 1

    # The width and height of original image is stored
    width2 = img.shape[1]
    height2 = img.shape[0]

    # Width Ratio and Height Ratio are calculated
    width_ratio = float(width1) / float(width2)
    height_ratio = float(height1) / float(height2)

    # A count variable is defined to keep track of number of elements stored in the array
    count = 0

    # The new image's array is initialised
    new = []

    # Every pixel of ther region of interest is traversed
    # We perform bilinear interpolation for the 3 channels (B,G,R) of the image
    for i in range((height2)):
        for j in range((width2)):
            x = int(width_ratio * j)
            y = int(height_ratio * i)
            x_diff = (width_ratio * j) - x
            y_diff = (height_ratio * i) - y

            # The neighbouring co-ordinates are compared
            # The 'if-else' conditions are added to check for the extreme co-ordinates
            if (x >= (width1 - 1) or y >= (height1 - 1)):
                A_blue = roi[y][x][0]
                A_red = roi[y][x][1]
                A_green = roi[y][x][2]
            else:
                A_blue = roi[y][x][0]
                A_red = roi[y][x][1]
                A_green = roi[y][x][2]

            if ((x + 1) >= (width1 - 1) or (y >= (height1 - 1))):
                B_blue = roi[y][x][0]
                B_red = roi[y][x][1]
                B_blue = roi[y][x][2]
            else:
                B_blue = roi[y + 1][x][0] & 0xff
                B_red = roi[y + 1][x][1]
                B_green = roi[y + 1][x][2]
            if (x >= (width1 - 1) or ((y + 1) >= (height1 - 1))):
                C_blue = roi[y][x][0]
                C_red = roi[y][x][1]
                C_green = roi[y][x][2]
            else:
                C_blue = roi[y][x + 1][0] & 0xff
                C_red = roi[y][x + 1][1]
                C_green = roi[y][x + 1][2]
            if ((x + 1) >= (width1 - 1) or (y + 1) >= (height1 - 1)):
                D_blue = roi[y][x][0] & 0xff
                D_red = roi[y][x][1]
                D_green = roi[y][x][2]
            else:
                D_blue = roi[y + 1][x + 1][0] & 0xff
                D_red = roi[y + 1][x + 1][1]
                D_green = roi[y + 1][x + 1][2]

            # Combining all the different channelsn into overall 3 channels
            newimg_blue = (int)((A_blue * (1 - x_diff) * (1 - y_diff)) + (B_blue * (x_diff) * (1 - y_diff)) + (
                        C_blue * (y_diff) * (1 - x_diff)) + (D_blue * (x_diff * y_diff)))
            newimg_red = (int)((A_red * (1 - x_diff) * (1 - y_diff)) + (B_red * (x_diff) * (1 - y_diff)) + (
                        C_red * (y_diff) * (1 - x_diff)) + (D_red * (x_diff * y_diff)))
            newimg_green = (int)((A_green * (1 - x_diff) * (1 - y_diff)) + (B_green * (x_diff) * (1 - y_diff)) + (
                        C_green * (y_diff) * (1 - x_diff)) + (D_green * (x_diff * y_diff)))

            # Adding the values into the array
            newrow = count // (width2)
            newcol = count % (width2)
            newimg = [newimg_blue, newimg_red, newimg_green]
            if (newcol == 0):
                new.append([])
            new[newrow].append(newimg)
            count += 1

    final_img = np.uint8(new)
    img_boxed = img.copy()
    cv2.rectangle(img_boxed, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 3)
    # Final Image is returned
    return final_img, img_boxed

#example usage of zoom function
# if __name__ == "__main__":
#     # Path of input image is given
#     path = '/Users/armonbarakchi/Desktop/ECE253_ArmonBarakchi_HenryPritchard/istockphoto-1149340384-612x612.jpg'
#     img = cv2.imread('/Users/armonbarakchi/Desktop/ECE253_ArmonBarakchi_HenryPritchard/istockphoto-1149340384-612x612.jpg')
#
#     # Pivot co-ordinates and scale are given
#     x_coord = 100
#     y_coord = 100
#     scale = 6.0
#
#     # Final image is got
#     org, final_img = zoom(path, x_coord, y_coord, scale)
#     cv2.imshow('Original', org)
#     cv2.imshow('Final', final_img)
#
#     # Zoomed image is saved as 'zoom_img.jpg'
#     #cv2.imwrite('zoom_img.jpg', final_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

