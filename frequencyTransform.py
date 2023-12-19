import numpy as np
import cv2 as cv

path1 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos000_000.ome.jpg"
path2 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos001_000.ome.jpg"
# path1 = R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\MIST_test\img_r000_c000.jpg"
# path2 = R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\MIST_test\img_r000_c001.jpg"



# img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)


def fft2d(inp):
    return np.fft.fft2(inp)

def ifft2d(inp):
    return np.abs(np.fft.ifft2(inp))


def phaseCorrelation(image1, image2):
    G_1 = fft2d(image1)
    G_2 = fft2d(image2)

    c = G_1*np.conj(G_2)

    d = ifft2d(c/np.abs(c))

    return np.abs(d)/np.amax(np.abs(d))

def phaseCorrelationOverlap(image1, image2, overlap, origin="LEFT"): 
    #TODO: introduce error
    if origin == "LEFT":
        inp_img1 = image1[int(len(image1)*(1-overlap/100)):, :]
        inp_img2 = image2[:int(len(image2)*overlap/100) +1, :]
    
    if origin == "RIGHT":
        inp_img1 = image1[:int(len(image2)*overlap/100)+1, :]
        inp_img2 = image2[int(len(image1)*(1-overlap/100)):, :]
    
    if origin == "BOTTOM":
        inp_img1 = image1[:, :int(len(image2)*overlap/100)+1]
        inp_img2 = image2[:, int(len(image1)*(1-overlap/100)):]

    if origin == "TOP":
        inp_img1 = image1[:, int(len(image2)*overlap/100):]
        inp_img2 = image2[:, :int(len(image1)*(1-overlap/100))+1]
    
    d = phaseCorrelation(inp_img1, inp_img2)

    displacement = np.unravel_index(d.argmax(), d.shape)
    return (displacement[1], displacement[0] + int(len(image1[0])*(1-overlap/100)))



a = np.vander((5, 4, 3, 2, 1), 4)

print(a)

d = np.zeros((5, 5), dtype=np.uint8)
c = np.ones(5, dtype=np.uint8)




img1_g = cv.GaussianBlur(img1, (11, 11), 0)
img2_g = cv.GaussianBlur(img2, (11, 11), 0)

# d = phaseCorrelationOverlap(img1, img2, 5)
# print(d.shape)
# print(np.unravel_index(d.argmax(), d.shape))
# displacement = np.unravel_index(d.argmax(), d.shape)

# displacement = (img1.shape[0] - displacement[0], img1.shape[1] - displacement[1])

displacement = phaseCorrelationOverlap(img1_g, img2_g, 5)

def stitchImgs(image1, image2, displacement):

    print(displacement)

    x = displacement[0]
    y = displacement[1]
    out = np.zeros((x + image1.shape[0], y + image1.shape[1]), dtype=image1.dtype)
    out[:image1.shape[0], :image1.shape[1]] += image1
    out[x:, y:] += image2

    return out


# d = 255*d/np.max(d)


cv.imshow("stitch", stitchImgs(img1, img2, displacement))
cv.waitKey(0)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

print(d.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# a, b, c = axes3d.get_test_data(0.05)

# print(np.all(a == b.T))
# X = np.tile(np.arange(0, d.shape[0],1, dtype=np.uint8), (d.shape[1], 1)) #TODO: repeat Y times


# Y = X.T

Z = d
# Grab some test data.

k = a - b.T
X, Y = np.meshgrid(np.linspace(0, d.shape[1], d.shape[1]), np.linspace(0, d.shape[0], d.shape[0]))
# Plot a basic wireframe.
ax.plot_surface(X, Y, Z, rstride=10, cstride=10)

plt.show()


