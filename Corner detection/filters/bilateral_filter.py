import cv2 as cv
import numpy as np


image = cv.imread(R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg", cv.IMREAD_GRAYSCALE)
image = cv.imread(R"C:\data\dsd_10\dsd_10\Out\img_stitch_bleng_vignettcorrection.png", cv.IMREAD_GRAYSCALE)

# Resize the filtered image to fit the window
height, width = image.shape[:2]
new_width = 800  # Set the desired width
scaling_factor = new_width / width
resized_image = cv.resize(image, (new_width, int(height * scaling_factor)))
filtered_image = cv.bilateralFilter(resized_image, d=9, sigmaColor=18, sigmaSpace=5)

gaussian_filtered_image = cv.GaussianBlur(resized_image, (9, 9), 0)


cv.imshow('Filtered Image', filtered_image)
cv.imshow('Filtered Image gauss', gaussian_filtered_image)
cv.waitKey(0)
cv.destroyAllWindows()

from hough_transform import *

doTransform(filtered_image)
