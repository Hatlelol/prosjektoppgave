
import numpy as np
import cv2 as cv
from tif import *

import matplotlib.pyplot as plt


def findHistogram(image, hist_size):
    
    
    hist_size = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False

    hist_w = 512
    hist_h = 400

    if len(image.shape) == 3:
        bgr_planes = cv.split(image)

        b_hist = cv.calcHist(bgr_planes, [0], None, [hist_size], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [hist_size], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [hist_size], histRange, accumulate=accumulate)
        cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        return b_hist, g_hist, r_hist

    else:
        hist = cv.calcHist(image, [0], None, [hist_size], histRange, accumulate=accumulate)

        cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)



        return hist

    # for i in range(1, histSize):
    #     cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
    #         ( bin_w*(i), hist_h - int(b_hist[i]) ),
    #         ( 255, 0, 0), thickness=2)


    # cv.imshow('calcHist Demo', histImage)
    # cv.waitKey()
    # cv.destroyAllWindows()


def drawHist(image, hist, color):
    hist_size = 256
    hist_w = image.shape[1]
    hist_h = image.shape[0]
    bin_w = int(round(hist_w/hist_size))
    for i in range(1, hist_size):
        cv.line(image, ( bin_w*(i-1), hist_h - int(hist[i-1]) ),
                    ( bin_w*(i), hist_h - int(hist[i]) ),
                    color, thickness=2)
    return image
import matplotlib.pyplot as plt
import cv2

im = cv2.imread(R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\Latex\prosjektoppgave\Images\Implementation\object detection\Fused_pic2_no_overlap_compute-jpg.jpg")
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Num")
plt.show()

# # filename = R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg"
# # filename = R"C:\data\grid\Fused_pic2_no_overlap_compute-1.tif"
# filename = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_000.ome.tif"
# img = read(filename, key=0)
# print(np.amax(img))
# img = (img*(255/np.amax(img))).astype(np.uint8)
# print(np.amax(img))

# # histSize = 256
# # histRange = (0, 256) # the upper boundary is exclusive
# # accumulate = False

# # b_hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)

# # hist_w = 512
# # hist_h = 400
# # bin_w = int(round( hist_w/histSize ))
# # histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
# # cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
# # for i in range(1, histSize):
# #  cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
# #  ( bin_w*(i), hist_h - int(b_hist[i]) ),
# #  ( 255, 0, 0), thickness=2)


# # cv.imshow('calcHist Demo', histImage)
# # cv.waitKey()
# # cv.destroyAllWindows()

# # Load the image
# # img = io.imread("../_static/figs/daisy.png", as_gray=True) * 255

# # Display the image
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# ax.imshow(img, cmap="gray")
# plt.show()

# # Prepare the figure for the histograms
# fig, axs = plt.subplots(1, 2, figsize=(12,4))

# # Plot the first histogram
# axs[0].hist(
#     img.ravel(),           # The image must be flattened to use function hist
#     bins=range(0,256,2)    # Define 128 bins between 0 and 255
# )
# axs[0].set_xlabel("Intensities")
# axs[0].set_ylabel("Number")

# # Plot the second histogram
# axs[1].hist(
#     img.ravel(),           # The image must be flattened to use function hist
#     bins=range(0,256,16)   # Define 16 bins between 0 and 255
# )
# axs[1].set_xlabel("Intensities")
# axs[1].set_ylabel("Number")

# # Show the figure
# plt.show()
# img[img < 30] = 0
# s = np.sum(img)
# s_blk = np.sum(img == 0)
# avg = s/(len(img.flatten()) - s_blk)
# print(avg)
# img[img > avg] = 255

# dst = cv.equalizeHist(img)

# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# ax.imshow(dst, cmap="gray")
# plt.show()

# # Prepare the figure for the histograms
# fig, axs = plt.subplots(1, 2, figsize=(12,4))

# # Plot the first histogram
# axs[0].hist(
#     dst.ravel(),           # The image must be flattened to use function hist
#     bins=range(0,256,2)    # Define 128 bins between 0 and 255
# )
# axs[0].set_xlabel("Intensities")
# axs[0].set_ylabel("Number")

# # Plot the second histogram
# axs[1].hist(
#     dst.ravel(),           # The image must be flattened to use function hist
#     bins=range(0,256,16)   # Define 16 bins between 0 and 255
# )
# axs[1].set_xlabel("Intensities")
# axs[1].set_ylabel("Number")

# # Show the figure
# plt.show()
