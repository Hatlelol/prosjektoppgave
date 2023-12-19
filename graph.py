import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# gray = cv.imread(R"C:\data\dsd_10\dsd_10\Out\img-stitched-0.jpg", cv.IMREAD_GRAYSCALE)
gray = cv.imread(R"C:\Users\sondr\OneDrive\Skrivebord\img-stitched-0.ome.ome.jpg", cv.IMREAD_GRAYSCALE)
# gray = cv.imread(R"C:\data\grid\Fused_pic2_no_overlap_compute.tif", cv.IMREAD_GRAYSCALE)

gray = gray[::4, ::4]
gray = gray[::2, ::2]
# gray = gray[::2, ::2]
gray = gray[1000:, :]
#  Edge detection
cv.imshow("gray image", gray)
# dst = cv.Canny(gray, 20, 200, None, 3, True)
dst1 = cv.Canny(gray, 40, 150, None, 3, True)
# dst2 = cv.Canny(gray, 40, 200, None, 3, True)
# dst3 = cv.Canny(gray, 40, 230, None, 3, True)

# cv.imshow("Canny Edge", dst)
cv.imshow("Canny Edge1", dst1)
# cv.imshow("Canny Edge2", dst2)
# cv.imshow("Canny Edge3", dst3)
cv.waitKey(0)

r = 1 #rho
t = np.pi/180 #theta
threshold = 100

lines = cv.HoughLines(dst1, r, t, threshold, None, 0, 0)

cdst = cv.cvtColor(dst1, cv.COLOR_GRAY2BGR)

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





# # for i in range(1, 10):

# #     linenum = i*int(len(dst)/10)
# #     line = dst[linenum]

# #     plt.plot(line)
# #     plt.title(f"Line number: {linenum}")
# #     plt.ylabel("Intensity")
# #     plt.show()


# line_avg = np.average(gray, axis=0)

# plt.plot(line_avg)
# plt.show()

# # derivate = np.diff(line_avg)

# # plt.plot(derivate)
# # plt.show()

# mag = np.fft.fft(line_avg)
# freq = np.fft.fftfreq(line_avg.shape[-1])
# plt.plot(freq, mag.real, freq, mag.imag)
# plt.show()

# abs_derivate = np.abs(derivate)

# spikes = abs_derivate > np.max(abs_derivate)*0.03
# print(spikes)

# plt.plot(abs_derivate)
# plt.plot(np.full(len(abs_derivate), np.average(abs_derivate)))
# plt.plot(np.full(len(abs_derivate), np.average(abs_derivate)))
# plt.show()


# conv = np.convolve(abs_derivate, np.array([0.7, 0.15, 0.15])[::-1], 'same')
# plt.plot(conv)

# plt.show()


# xdiff = line_avg[1:] - line_avg[0:-1]

# xdiff_mean = np.abs(xdiff).mean()

# # Identify all indices greater than the mean
# spikes = xdiff > abs(xdiff_mean)+1
# print(line_avg[1:][spikes])  # prints 50, 100, 80
# print(np.where(spikes)[0]+1)  # prints 3, 7, 15
# k = 0
# sp = np.where(spikes)[0]+1
# for i in range(len(sp)):
#     if i == 0:
#         continue
#     if sp[i] == sp[i - 1] + 1:
#         continue
#     k += 1
# print(k) 

# # plt.plot(line_avg)
# plt.title(f"line average")
# # plt.plot(np.diff(line_avg))
# plt.plot(np.where(np.abs(np.diff(line_avg)) < np.average(np.abs(np.diff(line_avg))), np.diff(line_avg), 0))

# # plt.plot(np.full(len(line_avg), np.mean(np.abs(np.diff(line_avg)))))
# plt.show()




# cv.waitKey(0)
# cv.destroyAllWindows()