import tifffile
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
img = tifffile.imread(R"C:\data\grid\Fused_pic2_no_overlap_compute-1.tif")


# for i in range(1, 10):

#     linenum = i*int(len(dst)/10)
#     line = dst[linenum]

#     plt.plot(line)
#     plt.title(f"Line number: {linenum}")
#     plt.ylabel("Intensity")
#     plt.show()


# sobel_x= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel_y= np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# from scipy.signal import convolve

# img_sobel_x = convolve(img, sobel_x, mode='same')
# img_sobel_y = convolve(img, sobel_y, mode='same')
# gradient_magnitude = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))

line_y = img[1000]
line_x = img[:, 100]

print(line_y[0])
# plt.plot(line_y)

img = cv.GaussianBlur(img, (5, 5), 0)
img = cv.GaussianBlur(img, (5, 5), 0)
img = cv.GaussianBlur(img, (5, 5), 0)
def getCenterArea(line):
    a = np.where(np.abs(line - np.average(line, axis=0)) > np.amax(line)/2)
    plt.plot(np.abs(line - np.average(line, axis=0)))
    plt.plot(np.average(line))
    b = np.diff(a)
    f = a[0][np.argmax(b)]
    t = a[0][np.argmax(b) + 1]
    plt.show()
    return f, t

# line_avg = np.average(line_y, axis=0)
# line_mean = np.mean(line_y, axis=0)

# t = np.abs(line_y - line_avg)
# a = np.where( t> np.amax(line_y)/4)
# f = a[0][np.argmax(np.diff(a))]
# print(a[0][np.argmax(np.diff(a))-1])
# t = a[0][np.argmax(np.diff(a))+1]
# plt.plot(line_y[a[0][np.argmax(np.diff(a))]:a[0][np.argmax(np.diff(a))+1]])
# # print(np.diff(a))
# # print(a)
y_f, y_t = getCenterArea(img[int(img.shape[0]/2)])
x_f, x_t = getCenterArea(img[:, int(img.shape[1]/2)])

# plt.plot(np.full(len(line_y), line_avg))
# plt.plot(np.full(len(line_y), line_mean))

img[y_f:y_t] = 0

# plt.imshow(img, cmap='gray')
plt.show()