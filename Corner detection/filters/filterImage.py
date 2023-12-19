import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

path = R"C:\data\The-original-cameraman-image_W640.jpg"
image = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.imshow("image", image)
cv.waitKey(0)


# #bilatteral filter
# filtered_image = cv.bilateralFilter(image, d=9, sigmaColor=18, sigmaSpace=5)
# cv.imshow("bilateral filter", filtered_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

#Gaussian filter


# filtered_image = cv.GaussianBlur(image, (9, 9), 0)
# cv.imshow("gausian filter", filtered_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
#histogram
# vals = image.flatten()
# # plot histogram with 255 bins
# b, bins, patches = plt.hist(vals, 255)
# plt.xlim([0,255])
# plt.title("Histogram")
# plt.xlabel("Intensity")
# plt.ylabel("Num")
# plt.show()


# canny_image = cv.Canny(image, 100, 200, None, 3, L2gradient=True)
# cv.imshow("canny", canny_image)
# cv.waitKey(0)
# template = cv.imread(R"C:\data\The-original-cameraman-image_W640_face.jpg", cv.IMREAD_GRAYSCALE)
# w, h = template.shape[::-1]

# cv.imshow("Corner", image)
# cv.imshow("Template", template)

# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED'] #, # 'cv.TM_CCORR',
#  #'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# print(image.shape)
# print(template.shape)
# for meth in methods:
#     img = image.copy()
#     method = eval(meth)
#     # Apply template Matching

#     res = cv.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)


#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc

#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

from matplotlib.pyplot import figure

fft = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
# fft = np.fft.fft2(img)
fshift = np.fft.fftshift(fft)
magnitude_spectrum = 20*np.log(cv.magnitude(fshift[:,:,0], fshift[:,:,1]))


figure(figsize=(10, 10), dpi=100)

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

hrow, hcol = int(image.shape[0]/2), int(image.shape[1]/2)

window_size = 25

mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
mask[hrow - 25: hrow + 25, hcol-window_size:hcol+window_size] = 1
fshift_filtered = fshift*mask
f_ishift = np.fft.ifftshift(fshift_filtered)
ifft = cv.idft(f_ishift)
image_back = cv.magnitude(ifft[:, :, 0], ifft[:, :, 1])

magnitude_filter = 20*np.log(cv.magnitude(fshift_filtered[:,:,0],fshift_filtered[:,:,1]))
image_back = image_back*(255/np.amax(image_back))
cv.imshow("back", image_back.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()

figure(figsize=(15, 15), dpi=200)

plt.subplot(121),plt.imshow(image_back, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_filter, cmap = 'gray')
plt.title('Magnitude spectrum, low pass filter'), plt.xticks([]), plt.yticks([])
plt.show()

window_size = 25

mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
mask[hrow - 25: hrow + 25, hcol-window_size:hcol+window_size] = 1
mask = np.abs(mask - 1)
fshift_filtered = fshift*mask
f_ishift = np.fft.ifftshift(fshift_filtered)
ifft = cv.idft(f_ishift)
image_back = cv.magnitude(ifft[:, :, 0], ifft[:, :, 1])

magnitude_filter = 20*np.log(cv.magnitude(fshift_filtered[:,:,0],fshift_filtered[:,:,1]))
image_back = image_back*(255/np.amax(image_back))
cv.imshow("back", image_back.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()

figure(figsize=(15, 15), dpi=200)

plt.subplot(121),plt.imshow(image_back, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_filter, cmap = 'gray')
plt.title('Magnitude spectrum, high pass filter'), plt.xticks([]), plt.yticks([])
plt.show()