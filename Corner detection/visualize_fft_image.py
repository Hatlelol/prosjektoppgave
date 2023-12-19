

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import cv2 as cv

def applyNoise(rng: np.random.Generator, image: np.ndarray, vignett:bool = False, gaussian: bool = False, sAndp: bool = False, speckle: bool = False, poisson: bool = False) -> (np.ndarray, dict):
    """
    Add noise by the modes listed. Adds them in a way that is natural for image aquisition.
    set each if the parameters to True to add the noise from them

    poisson noise is not fully operational
    
    """
    
    noise_types = {
        "specke": speckle,
        "vignett": vignett,
        "gaussian": gaussian,
        "poisson": poisson,
        "salt and peppet": sAndp
    }
    
    
    # rng = np.random.default_rng(42)
    if speckle:
    
        gauss = 0.07*np.random.randn(*image.shape)
        gauss = gauss.reshape(*image.shape)      
        image = np.where(image + (image * gauss).astype(np.uint8)>255, 255, image + (image * gauss).astype(np.uint8)) 

    if vignett:
        X_resultant_kernel = cv.getGaussianKernel(image.shape[1],image.shape[1])
        Y_resultant_kernel = cv.getGaussianKernel(image.shape[0],image.shape[0])
        
        #generating resultant_kernel matrix 
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

        #creating mask and normalising by using np.linalg
        # function
        mask = resultant_kernel / np.amax(resultant_kernel)

        image = (image*mask).astype(np.uint8)
        # image = (255/np.amax(image))*image
    if gaussian:
        gauss = rng.normal(0, np.sqrt(255*0.5), image.shape)
        image = np.where(image + gauss > 255, image - gauss, (image + gauss)).astype(np.uint8)
   
    if poisson:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = (rng.poisson(image * vals) / float(vals)).astype(np.uint8)
    if sAndp:
        s_vs_p = 0.5
        amount = 0.005
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        y_coords = np.random.randint(0, image.shape[0]-1, int(num_salt))
        x_coords = np.random.randint(0, image.shape[1]-1, int(num_salt))

        image[y_coords, x_coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        y_coords = np.random.randint(0, image.shape[0]-1, int(num_pepper))
        x_coords = np.random.randint(0, image.shape[1]-1, int(num_pepper))
    
        image[y_coords, x_coords] = 0
    return image, noise_types

# Function to plot the 2D Fourier transform
def plot_2d_fourier_transform(image, dx, dy):
    fft_result = fft2(image)
    fft_result_shifted = fftshift(fft_result)

    freq_x = np.fft.fftshift(np.fft.fftfreq(image.shape[0], dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(image.shape[1], dy))

    fft_alter = np.angle(fft_result_shifted)

    ifft = np.abs(np.fft.ifft2(np.fft.fft2(fft_alter)))


    # plt.imshow(np.angle(fft_result_shifted/np.abs(fft_result_shifted)), extent=(freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()))
    plt.imshow(ifft)
    plt.colorbar()
    plt.title('2D Fourier Transform of Image')
    plt.xlabel('Frequency (cycles per unit length)')
    plt.ylabel('Frequency (cycles per unit length)')
    plt.show()

# Parameters
image_size = 256  # Size of the 2D image
pixel_size = 0.1  # Size of each pixel in the image (adjust based on your specific image)

# Generate 2D image
# image = generate_image(image_size)
image = cv.imread(R"C:\data\The-original-cameraman-image_W640.jpg", cv.IMREAD_GRAYSCALE)
rng = np.random.default_rng()
# image, _ = applyNoise(rng, image, vignett=True, gaussian=True, sAndp=True, speckle=True)
shape = image.shape
if shape[0] != shape[1]:
    print(shape)
    if shape[0] > shape[1]:
        image = image[:shape[1], :]
    else:
        image = image[:, :shape[2]]
shape = image.shape
shift = 200
image1 = image[:shape[0]-shift, :shape[1]-shift]
image1, _ = applyNoise(rng, image1, vignett=True, gaussian=True, sAndp=True, speckle=True)
image2 = image[shift:, shift:]
image2, _ = applyNoise(rng, image2, vignett=True, gaussian=True, sAndp=True, speckle=True)
print(image1.shape, image2.shape)

F = np.fft.fft2(image1)
G = np.fft.fft2(image2)


res = F*np.conj(G)
res = res#/np.abs(res)
ifftres = np.fft.ifft2(res).real.astype(np.float32)
plt.imshow(ifftres/np.amax(ifftres), aspect="auto")#, cmap="gray")#/(np.amax(np.abs(ifftres))))
plt.title(R"$|\mathcal{F}^{-1}\{F(u, v)^\ast \cdot G(u, v)\}|$")
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
cbar = plt.colorbar()
cbar.set_label("Normalized intensity (a. u)")
plt.show()

gamma = np.fft.ifft2(res/np.abs(res)).real.astype(np.float32)

plt.imshow(gamma, aspect="auto")#/(np.amax(np.abs(ifftres))))
plt.title(R"$|\mathcal{F}^{-1}\{\frac{F(u, v)^\ast \cdot G(u, v)}{|F(u, v)^\ast \cdot xG(u, v)|}\}|$")
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
cbar = plt.colorbar()
cbar.set_label("Normalized intensity (a. u)")
plt.show()
shape = gamma.shape
position = [200, 200]
f = 10
# for f in range(10):
x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
distances = np.sqrt((x - position[1])**2 + (y - position[0])**2)
distances = np.exp(-f*(distances/np.amax(distances))**2)
correlation = gamma*distances

plt.imshow(correlation, aspect="auto")#/(np.amax(np.abs(ifftres))))
plt.title(R"$\gamma'(x, y)$")
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
cbar = plt.colorbar()
cbar.set_label("Normalized intensity (a. u)")
plt.show()
    # image[:shape[0]-shift, :shape[1]-shift] = image1
# image[shift:, shift:] = image2

image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

image[-shift - 1: -shift +1, :-shift] = (0, 0, 255)
image[:-shift, -shift - 1: -shift +1] = (0, 0, 255)
image[:-shift, :4] = (0, 0, 255)
image[:4, :-shift] = (0, 0, 255)
image = cv.putText(image, 'Image1', (125, 175), cv.FONT_HERSHEY_SIMPLEX ,  1, (0, 0, 255), 2, cv.LINE_AA) 
image[shift - 1: shift +1, shift:] = (255, 0, 0)
image[shift:, shift - 1: shift +1] = (255, 0, 0)
image[shift:, -2:] = (255, 0, 0)
image[-2:, shift:] = (255, 0, 0)
image = cv.putText(image, 'Image2', (shape[0] - 200, shape[0] - 180), cv.FONT_HERSHEY_SIMPLEX ,  1, (255, 0, 0), 2, cv.LINE_AA) 
plt.imshow(image)
plt.show()
plt.imshow(image1, cmap="gray")
plt.show()
plt.imshow(image2, cmap="gray")
plt.show()
# Plot the original image
# plt.imshow(image, extent=(-5, 5, -5, 5), cmap='gray')
# plt.title('Original Image')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.colorbar()
# plt.show()

# # Plot the 2D Fourier transform
# plot_2d_fourier_transform(image1, pixel_size, pixel_size)
# plot_2d_fourier_transform(image2, pixel_size, pixel_size)
# # plt.show()