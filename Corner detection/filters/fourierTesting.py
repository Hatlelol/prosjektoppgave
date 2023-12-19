import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


full = R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg"
full = R"C:\data\grid\Fused_pic2_no_overlap_compute-1.tif"
img = cv.imread(full, cv.IMREAD_GRAYSCALE)[::2, ::2]


fft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
# fft = np.fft.fft2(img)
fshift = np.fft.fftshift(fft)
magnitude_spectrum = 20*np.log(cv.magnitude(fshift[:,:,0], fshift[:,:,1]))


figure(figsize=(10, 10), dpi=100)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

hrow, hcol = int(img.shape[0]/2), int(img.shape[1]/2)

window_size = 200

mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
mask[hrow - 50: hrow + 50, hcol-window_size:hcol+window_size] = 1
fshift_filtered = fshift*mask
f_ishift = np.fft.ifftshift(fshift_filtered)
ifft = cv.idft(f_ishift)
img_back = cv.magnitude(ifft[:, :, 0], ifft[:, :, 1])

magnitude_filter = 20*np.log(cv.magnitude(fshift_filtered[:,:,0],fshift_filtered[:,:,1]))
img_back = img_back*(255/np.amax(img_back))
cv.imshow("back", img_back.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()

figure(figsize=(15, 15), dpi=200)

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_filter, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()