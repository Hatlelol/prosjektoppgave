import numpy as np
import cv2 as cv
from verification.generateTest import *
from filters.hough_transform import *
import matplotlib.pyplot as plt

# Example usage:
N = 1000
M = 1000

image_base = np.full((N, M, 3), (126, 125, 126), dtype=np.uint8)

image = image_base.copy() 

image, image2 = generateAndDrawRectangles(image, thickness=3, for_hough=True)
cv.imshow("img", image)
cv.waitKey(0)

rectangles = np.where(image2 < 100)


print(rectangles)
thetas = np.linspace(-np.pi/2, np.pi /2, 2000)
image = np.zeros((2000, len(thetas)), dtype=np.uint32)
max_rho = 0
min_rho = 0
for y, x in zip(rectangles[0], rectangles[1]):
    rho = x*np.cos(thetas) + y*np.sin(thetas)
    max_rho = np.amax(rho) if np.amax(rho) > max_rho else max_rho
    min_rho = np.amin(rho) if np.amin(rho) < min_rho else min_rho
    # print(np.amax(rho), np.amin(rho))

    image[(rho).astype(int), np.arange(0, len(thetas), 1, int)] += 1

    # for i in range(len(rho)):
    #     image[int(rho[i]) + 1000, i] += 1


# floor_mask = np.logical_and(image > 0, image < 255)
# image[floor_mask] = image[floor_mask]/2
# ceil_mask = image >= 255
# image[ceil_mask] = (image[ceil_mask] - 255/2)/(np.amax(image[ceil_mask]) - 255)




rollamound = abs(min_rho) if abs(min_rho) + max_rho < 2000 else 0
print(rollamound)
image = np.roll(image, int(abs(min_rho)), axis=0)

dst = cv.equalizeHist(image.astype(np.uint8))
# image = np.sqrt(image)/2
print(np.amax(image))

# Plot Hough space
plt.imshow(dst, extent=[thetas.min(), thetas.max(), -1000, 1000], aspect='auto', cmap='gray')
plt.colorbar(label='Correction magnitude')
plt.title('Hough Space')
plt.xlabel('Theta (radians)')
plt.ylabel('Rho')
plt.show()


