# """
# Implementation based on the paper by wang et al: Image quality assessment: from error visibility to structural similarity
# https://ieeexplore.ieee.org/document/1284395
# C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\papers\Image_quality_assessment_from_error_visibility_to_structural_similarity.pdf
# """
import numpy as np
import cv2
from pathlib import Path





### Structural similarity based image quality assessment:


def luminance(mu_x, mu_y, C_1):
    # Luminance comparison, conensiding with webers law 
    return (2*mu_x*mu_y + C_1)/(mu_x**2 + mu_y**2 + C_1)


def contrast(sig_x, sig_y, C_2):
    return (2*sig_x*sig_y + C_2)/(sig_x**2 + sig_y**2 + C_2)


def correlation(x, mu_x, y, mu_y, w):
    return np.sum(w*(x - mu_x)*(y - mu_y))/(len(x) - 1)


def structure(corr, sig_x, sig_y, C_3):
    return (corr + C_3)/(sig_x*sig_y + C_3)

def SSIM_simplified(corr, sig_x, sig_y, mu_x, mu_y, C_1, C_2):
    return (2*mu_x*mu_y + C_1)*(2*corr + C_2)/((mu_x**2 + mu_y**2 + C_1)*(sig_x**2 + sig_y**2 + C_2))

def SSIM(l, c, s, alpha, betta, yotta):
    return (l**alpha) * (c**betta) * (s**yotta)


def computeSSIM(img_x, img_y, kernel, C_1, C_2):
    x = img_x.flatten()
    y = img_y.flatten()
    w = kernel.flatten()
    mu_x = np.sum(x * w)
    mu_y = np.sum(x * w)
    sig_x = np.sqrt(np.sum(w * np.power(x - mu_x, 2)))
    sig_y = np.sqrt(np.sum(w * np.power(y - mu_y, 2)))
    corr = np.sum(w*(x - mu_x)*(y - mu_y))

    # SSIM = SSIM(l, c, s, alpha, betta, yotta)
    return SSIM_simplified(corr, sig_x, sig_y, mu_x, mu_y, C_1, C_2)


# def main():
#     ### CONSTANTS BEGIN

#     K_1 = 0.01
#     K_2 = 0.03
#     K_3 = 0.01
#     L = 255 # TODO: needs to change based on image, dynamic range of pixel
#     alpha = 1
#     betta = 1
#     yotta = 1

#     ### CONSTANTS END



#     x = []
#     y = []

#     mu_x = np.mean(x)
#     mu_y = np.mean(y)
#     C_1 = (K_1*L)**2
#     C_2 = (K_2*L)**2
#     C_3 = C_2/2

#     sig_x = np.std(x, ddof=1)
#     sig_y = np.std(y, ddof=1)
#     corr = correlation(x, mu_x, y, mu_y)

#     # l = luminance(mu_x, mu_y, C_1)
#     # c = contrast(sig_x, sig_y, C_2)
#     # s = structure(corr, sig_x, sig_y, C_3)

#     # SSIM = SSIM(l, c, s, alpha, betta, yotta)
#     SSIM = SSIM_simplified(corr, sig_x, sig_y, mu_x, mu_y, C_1, C_2)


def getKernel(size=11, dtype=float, sigma=1.5, type="Gaussian"):
    if type == "Gaussian" or type is None:
        
        kernel = np.zeros((size + 2, size + 2), dtype=dtype)
        kernel[int(size/2) + 1, int(size/2) + 1] = 1
        kernel = cv2.GaussianBlur(kernel, (size, size), sigma)
        return kernel[1:-1, 1:-1]
    
    else: 
        print("No other kernel defined yet")


def compute_MSSIM(img_x, img_y, type="FULL", L=255):
    # Computing the MSSIM based on the two input images. Assumes that they are allinged in [0, 0]
    K_1 = 0.01
    K_2 = 0.03
    C_1 = (K_1*L)**2
    C_2 = (K_2*L)**2

    if img_x.shape[0] < img_y.shape[0]:
        img_y = np.pad(img_y, ((0, 0), (0, img_x.shape[0])), "constant")
    elif img_x.shape[0] > img_y.shape[0]:
        img_x = np.pad(img_x, ((0, 0), (0, img_y.shape[0])), "constant")
    if img_x.shape[1] < img_y.shape[1]:
        img_y = np.pad(img_y, ((0, img_x.shape[1]), (0, 0)), "constant")
    elif img_x.shape[1] > img_y.shape[1]:
        img_x = np.pad(img_x, ((0, img_y.shape[1]), (0, 0)), "constant")

    if img_x.shape != img_y.shape:
        print(f"Something went wrong with the padding of the arrays, x: {img_x.shape}, y: {img_y.shape}")
        return 10

    kernel = getKernel()
    if type == "FULL":
        MSSIM = 0
        for i in range(img_x.shape[0] - kernel.shape[0]):
            for j in range(img_x.shape[1] - kernel.shape[1]):
                MSSIM += computeSSIM(img_x[i:i+kernel.shape[0], j:j+kernel.shape[1]], img_y[i:i+kernel.shape[0], j:j+kernel.shape[1]], kernel, C_1, C_2)

        return MSSIM/((img_x.shape[0] - kernel.shape[0])*(img_x.shape[1] - kernel.shape[1]))
    if type == "KERNEL":
        MSSIM = 0
        for i in range(int(img_x.shape[0]/kernel.shape[0])):
            for j in range(int(img_x.shape[1]/kernel.shape[1])):
                x = img_x[i*kernel.shape[0]:(i+1)*kernel.shape[0], j*kernel.shape[1]:(j+1)*kernel.shape[1]]
                y = img_y[i*kernel.shape[0]:(i+1)*kernel.shape[0], j*kernel.shape[1]:(j+1)*kernel.shape[1]]
                MSSIM += computeSSIM(x, y, kernel, C_1, C_2)
        return MSSIM/(int(img_x.shape[0]/kernel.shape[0])*int(img_x.shape[1]/kernel.shape[1]))
    if type == "FOUR":
        MSSIM = 0
        oneFourth = (int(img_x.shape[0]/4), int(img_x.shape[1]/4))
        for i in [1, 3]:
            for j in [1, 3]:
                x = img_x[oneFourth[0]*i:oneFourth[0]*i + kernel.shape[0], oneFourth[1]*j:oneFourth[1]*j + kernel.shape[1]]
                y = img_y[oneFourth[0]*i:oneFourth[0]*i + kernel.shape[0], oneFourth[1]*j:oneFourth[1]*j + kernel.shape[1]]
                MSSIM += computeSSIM(x, y, kernel, C_1, C_2)
        return MSSIM/4

    print(f"Type {type} not implemented, try 'KERNEL' or 'FULL'")
    return 10




img_x = cv2.imread(R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\c++\projects\opencvtest\lenna.png", cv2.IMREAD_GRAYSCALE)
img_y = img_x.copy()
import time

start = time.time()
print(compute_MSSIM(img_x, img_y, type="FULL"))
print(f"Time elapsed for full equal image MSSIM: {time.time() - start}")
start = time.time()
print(compute_MSSIM(img_x, img_y, type="KERNEL"))
print(f"Time elapsed for KERNEL equal image MSSIM: {time.time() - start}")
start = time.time()
print(compute_MSSIM(img_x, img_y, type="FOUR"))
print(f"Time elapsed for FOUR equal image MSSIM: {time.time() - start}")

img_y = cv2.GaussianBlur(img_x, (15, 15), 0)
print()
print("IMG y is gaussian:")
print()
start = time.time()
print(compute_MSSIM(img_x, img_y, type="FULL"))
print(f"Time elapsed for full equal image MSSIM: {time.time() - start}")
start = time.time()
print(compute_MSSIM(img_x, img_y, type="KERNEL"))
print(f"Time elapsed for KERNEL equal image MSSIM: {time.time() - start}")
start = time.time()
print(compute_MSSIM(img_x, img_y, type="FOUR"))
print(f"Time elapsed for FOUR equal image MSSIM: {time.time() - start}")