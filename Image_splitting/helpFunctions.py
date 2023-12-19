import numpy as np

global UP
global RIGHT
UP = 0
RIGHT = 1

def splitImage(image:np.ndarray, overlap_w: int, overlap_h: int, rows: int, cols: int, shift_w: int, shift_h: int, zero_coord = "LL") -> np.ndarray:
    
    # TODO: add random shifts in placement of center, and pad with zeros
    pad_n = min(0, shift_h*(rows-1))
    pad_w = max(0, shift_w*(cols-1))
    pad_s = max(0, shift_h*(rows-1))
    pad_e = min(0, shift_w*(cols-1))

    w = image.shape[1]
    h = image.shape[0]

    image = np.pad(image, ((pad_n, pad_s), (pad_w, pad_e)), 'constant', constant_values=(0, 0))

    # pad image with zeros according to shift
    output_w = int(w/cols + (overlap_w/cols)*(cols-1))
    output_h = int(h/rows + (overlap_h/rows)*(rows-1))

    center = (int(output_h/2), int(output_w/2))

    tiles = np.zeros((rows, cols, output_h, output_w), dtype=image.dtype)
    coordinates = np.zeros((rows, cols, 2), int)
    coordinates_relative = np.zeros((rows, cols, 2, 2), int)

    for i in range(rows):
        y_start = abs(min(shift_w*rows, 0)) + output_h*i - overlap_h*i
        y_end = y_start + output_h
        for j in range(cols):
            x_start = abs(min(shift_h*cols, 0 )) + output_w*j + shift_h*i - overlap_w*j
            x_end = x_start + output_w
            print(f"{i}, {j}, {image[y_start + shift_w*j:y_end+shift_w*j, x_start:x_end].shape}, {output_h*i}, {output_w*j}, {y_start}, {x_start}")
            
            if i == 0 and j == 0:
                y_offset = y_start
                x_offset = x_start

            coordinates[i, j] = [y_start + shift_w*j - y_offset, x_start - x_offset]
            
            if i != rows - 1:
                coordinates_relative[i + 1, j][UP] = [shift_w, shift_h]  
            
            if j != cols - 1:
                coordinates_relative[i, j + 1][RIGHT] = [shift_w,  shift_w]

            
            tile = image[y_start+shift_w*j:y_end+shift_w*j, x_start:x_end]
            
            # TODO: apply noise

            tiles[i, j] = tile

    return tiles, coordinates, coordinates_relative








def trimInput(img_1: np.ndarray, img_2: np.ndarray, overlap: int, direction="LEFT"):
    """
    Returns only the overlapping regions of the input images img_1 and img_2
    
    img_1: Grayscale image 2d array
    img_2: Grayscale image 2d array
    overlap: maximum overlap between the images in pixels
    direction: the orientation of  img_1 wrt img_2

    returns: two arrays with either height or width scaled to be overlap + max_error percent of original    
    """
    if direction == "RIGHT":
        return img_1[:, :overlap],\
               img_2[:, img_1.shape[1]-overlap]
    if direction == "LEFT":
        return img_1[:, img_1.shape[1]-overlap:],\
               img_2[:, :overlap]
    if direction == "UP":
        return img_1[img_1.shape[0]-overlap:, :],\
               img_2[:overlap, :]
    if direction == "DOWN" or direction == "UNDER":
        return img_1[:overlap, :],\
               img_2[img_1.shape[0]-overlap:, :]
    print(f"Direction {direction} is not implemented yet.")

def divide_zero(dividend, divisor):

    return np.where(divisor == 0, 0, dividend/divisor)


def fft2d(inp):
    return np.fft.fft2(inp)

def ifft2d(inp):
    return np.abs(np.fft.ifft2(inp))

def phaseCorrelation(image1, image2):
    """
    Computes the phase correlation between the input images. 

    image1: NxM input array 
    image2: NxM input array

    return: the inverse fourier transform of the phases of image1 and image2 subtracted.     
    """

    if image1.shape != image2.shape:
        print(f"Image shapes needs to be equal, base: {image1.shape}, cmp: {image2.shape}")
        return

    G_1 = fft2d(image1)
    G_2 = fft2d(image2)
    c = G_1*np.conj(G_2)

    #TODO: add butterworth

    # try:
    d = ifft2d(divide_zero(c, np.abs(c)))
    # except RuntimeWarning:
    #     print(c)
    #     print(abs(c))
    return np.abs(d)/np.amax(np.abs(d))