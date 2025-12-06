from skimage.metrics import structural_similarity as ssim
import cv2
from GraphingFunctions import *


dct_img = cv2.imread("dct.png")
original = cv2.imread("test.jpg")
db2 = cv2.imread("db2 (2).png")
noisy = cv2.imread("noisy.png")

plotGraph([dct_img], [noisy], [original], "out.pdf")