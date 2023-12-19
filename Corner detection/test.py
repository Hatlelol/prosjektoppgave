import numpy as np
import cv2 as cv

from verification.generateTest import generateAndDrawRectangles


img = np.full((1000, 1000), 126, np.uint8)
img = generateAndDrawRectangles(img, 1.5, 0, 2)
gray = np.float32(img)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# img = img[::8, ::8]
for num in np.arange(0.15, 0.25, 0.025):
    dst = cv.cornerHarris(gray,50,29, num)
    out = img.copy()



    ret, dst = cv.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    out[dst>0.1*dst.max()] = [0, 0, 255]
    for i in range(1, len(corners)):
        print(corners[i])
        c = np.int32(corners[i])[::-1]
        out[c[0]-2:c[0]+2, c[1]-2:c[1]+2] = [255, 0, 0]


    #between 0.2 and 0.1 found best
    cv.imshow("dst", out)
    cv.waitKey(0)


