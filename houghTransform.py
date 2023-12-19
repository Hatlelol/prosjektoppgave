import sys
import math
import cv2 as cv
import numpy as np


gray = cv.imread(R"C:\data\dsd_10\dsd_10\Out\img-stitched-0.jpg", cv.IMREAD_GRAYSCALE)
# gray = cv.imread(R"C:\data\grid\Fused_pic2_no_overlap_compute.tif", cv.IMREAD_GRAYSCALE)
gray = gray[::4, ::4]
# gray = gray[1000:1500, :]
gray = gray[::2, ::2]
# gray = gray[::4, ::4]
 # Edge detection





cv.imshow("gray image", gray)
dst = cv.Canny(gray, 50, 100, None, 3)

top = 0
topFound = False
bot = len(gray[:, 0]) - 1
botFound = False
left = 0
leftFound = False
right = len(gray[0, :]) - 1
rightFound = False
while True:
    print(f"{top} {bot} {left} {right}")
    if topFound or np.all(dst[top, left:right] > 200):
        topFound == True
    else:
        top += 10
    if botFound or np.all(dst[bot, left:right] > 200):
        botFound == True
    else:
        bot -= 10
    if leftFound or np.all(dst[top:bot, left] > 200):
        leftFound == True
    else:
        left += 10
    if rightFound or np.all(dst[top:bot, right] > 200):
        rightFound == True
    else:
        right -= 10

    if rightFound and leftFound and botFound and topFound:
        dst = dst[top:bot, left:right]
        break
    
    if left > right or top > bot:
        print("Wraparound")
        break



cv.imshow("Canny Edge", dst)

 # Standard Hough Line Transform
r = 1 #rho
t = np.pi/180 #theta
threshold = 100

lines = cv.HoughLines(dst, r, t, threshold, None, 0, 0)

cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

# Draw the lines

def xInThreshold(x, thresh, arr):
    if len(arr) == 0:
        return False
    for el in arr:
        if abs(x - el) < thresh: 
            return True
    return False


x_coords = []

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        if a < np.pi/4:
            continue
        b = math.sin(theta)
        # print(b)
        x0 = a * rho
        if xInThreshold(x0, 20, x_coords):
            continue
        x_coords.append(x0)
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

print(len(x_coords))

cv.imshow("Detected lines", cdst)
cv.waitKey(0)
cv.destroyAllWindows()