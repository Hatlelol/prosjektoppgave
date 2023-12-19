import numpy as np
import cv2 as cv
from histogram import findHistogram, drawHist
from tif import *
from hough_transform import doTransform

debug = True
#images:
lowerLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_000.ome.tif"
upperLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_019.ome.tif"
upperRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_019.ome.tif"
lowerRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_000.ome.tif"
lowerLeft_n = R"C:\data\dsd_10\dsd_10\dsd_10_MMStack_1-Pos000_000.ome.tif"
upperLeft_n = R"C:\data\dsd_10\dsd_10\dsd_10_MMStack_1-Pos000_014.ome.tif"
full_jpg = R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg"
full_tif = R"C:\data\grid\Fused_pic2_no_overlap_compute-1.tif"
#try to do the full filtering and corner detection of a file:

# half_1_jpg = R"C:\data\dsd_10\dsd_10\Out\img-stitched-0.jpg"

image = cv.imread(full_tif, cv.IMREAD_GRAYSCALE)
image = (image/(255/np.amax(image))).astype(np.uint8)

#know that the image is in grayscale
# image = readAndNorm(full_tif, 0, 255)

hist = findHistogram(image, 256)

if debug:

    hist_w = 512
    hist_h = 400
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
 
    hist_img = drawHist(hist_img, hist, (255, 0, 0))
    cv.imshow("img", hist_img)
    cv.waitKey(0)




height, width = image.shape[:2]
new_width = 800  # Set the desired width
scaling_factor = new_width / width
resized_image = cv.resize(image, (new_width, int(height * scaling_factor)))

resized_image[resized_image > 200] = 255
resized_image[resized_image < 50] = 0

cv.imshow("norm", resized_image)
cv.waitKey(0)

canny, sHough, PHough = doTransform(resized_image, 100, 200)

cv.imshow("canny_1", canny)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform 1", sHough)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform 1", PHough)
cv.waitKey(0)

belatteral_filter_image = cv.bilateralFilter(resized_image, d=9, sigmaColor=18, sigmaSpace=5)

cv.imshow("bilatFilter", belatteral_filter_image)
cv.waitKey(0)

canny, sHough, PHough = doTransform(belatteral_filter_image, 100, 200)

cv.imshow("canny", canny)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", sHough)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", PHough)
cv.waitKey(0)


# resized_image = cv.equalizeHist(resized_image)

# cv.imshow("eqHist", resized_image)
# cv.waitKey(0)


mask = np.where(resized_image > 7*int(255/10), 254, 0).astype(np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_DILATE, (21, 21))
# print(kernel)
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# dilation = cv.dilate(mask, kernel, iterations = 1)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)
img_c = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)

img_c[closing > 250] = (255, 255, 255)






cv.imshow("norm", img_c)
cv.waitKey(0)

canny, sHough, PHough = doTransform(img_c, 100, 200)

cv.imshow("canny", canny)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", sHough)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", PHough)
cv.waitKey(0)


cv.destroyAllWindows()