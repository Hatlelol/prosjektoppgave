import numpy as np

import cv2 as cv

import os

def imread(path: str) -> np.ndarray:
    """
    Reads the image from path. It automaticaly reads in grayscale, and if the image is in tiff format it checks and normalizes the values to lay inbetween 0 and 255

    path: The path from where to read the image

    returns: A numpy array of dtype np.uint8 of the image
    
    """

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if np.amax(img < 200) and (path.endswith(".tif") or path.endswith(".tiff")):
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = ((255/np.amax(img))*img).astype(np.uint8)
    
    return img

def grayToBGR(img: np.ndarray) -> np.ndarray:
    """
    Converts the image from grayscale to bgr
    """
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
def BGRToGray(img: np.ndarray) -> np.ndarray:
    """
    Converts the input image from BGR to gray
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def imshow(text: str, image:np.ndarray, destroy: bool=False) -> None:
    """
    Shows the input image with the input text with opencv
    
    text: The text to be displayed in the window
    image: numpyarray of the image
    destroy (false): Bool to say if cv.destroyAllWindows() should be run after the image is shown
    
    """

    h, w = image.shape[0:2]
    neww = 1000
    newh = int(neww*(h/w))
    image = cv.resize(image.copy(), (neww, newh))
    cv.imshow(text, image)
    cv.waitKey(0)
    if destroy:
        cv.destroyAllWindows()


def getOutputDimensions(coordinates: list or np.ndarray or tuple, shape: list or np.ndarray or tuple) -> list:
    """
    Function that finds the maximum and minimum x and y values of the indecies. 
    
    coordinates: list of the coordinates of the images
    shape: the shape of the images

    returns:
    x_maks: maximum coordinate in the x axis, only positive
    x_min: minimum coordinate in the x axis, only negative or 0
    y_maks: maximum coordinate in the y axis, only positive
    y_min: minimum coordinate in the y axis, only negative or 0


    """
    x_maks = x_min = y_maks = y_min = 0
    for c in coordinates:
        
        y = c[0] if c[0] != np.inf else 0
        x = c[1] if c[1] != np.inf else 0
        if x + shape[1] > x_maks:
            x_maks = x + shape[1]
        if x < x_min:
            x_min = x
        if y + shape[0] > y_maks:
            y_maks = y + shape[0]
        if y < y_min:
            y_min = y
    return x_maks, x_min, y_maks, y_min


def genDistance(shape: tuple or list or np.ndarray) -> np.ndarray:
    """
    Generates a ndarray with shape = shape where each pixel has the distance to the closest edge as its value
    This function is based on a solution from chatgpt

    shape: the shape of the output image

    returns: A np ndarray of shape where each pixels value is the distance to the closest edge. 
    """
    if shape[0] % 2 != 0:
        y_ax_base = np.arange(0, int(shape[0]/2) + 1)
        y_coords = np.concatenate((y_ax_base, y_ax_base[:-1:-1]))
    else:
        y_ax_base = np.arange(0, int(shape[0]/2))
        y_coords = np.concatenate((y_ax_base, y_ax_base[::-1]))
    if shape[1] % 2 != 0:
        x_ax_base = np.arange(0, int(shape[1]/2) + 1)
        x_coords = np.concatenate((x_ax_base, x_ax_base[:-1:-1]))
    else:
        x_ax_base = np.arange(0, int(shape[1]/2))
        x_coords = np.concatenate((x_ax_base, x_ax_base[::-1]))

    x, y, = np.meshgrid(x_coords, y_coords)

    distance = np.minimum(x, y).astype(np.uint32)
    return distance


def generateOutput(images, coordinates, distance=False, vignett_inv=None, border = False):
    """
    Function for generating full mosaeic from images and coordinates. 

    Images: A 1d array of equal size images
    coordinates: a 1d array of coordinates for each image in images
    distance: Bool to tell if blending should be done by distance, or no blending at all.
    vingett_inv: array of the inverted vignette filter
    border: bool to tell if a border should be added in the stitched regions

    return: A full mosaeic. 

    
    """
    
    shape = images[0].shape

    x_maks, x_min, y_maks, y_min = getOutputDimensions(coordinates, shape)

    out = np.zeros((abs(y_min) + y_maks, abs(x_min) + x_maks), dtype=np.uint32)
    print(out.shape)
    d_v = None
    if distance:
        out_d = out.copy()
        if vignett_inv is None:
            d_v = genDistance(shape)
        else:
            d_v = genDistance(shape)*vignett_inv
        

    for c, img in zip(coordinates, images):
        dy = c[0] if c[0] != np.inf else 0
        dx = c[1] if c[1] != np.inf else 0
        f_y = y_maks - dy - shape[0]
        f_x = abs(x_min) + dx
        # print(dx, dy)
        print(f_x, f_y)
        if d_v is None:
            out[f_y:f_y + shape[0], f_x:f_x + shape[1]] = (img).astype(np.uint32)
        else:
            out[f_y:f_y + shape[0], f_x:f_x + shape[1]] += (img*d_v).astype(np.uint32)
            out_d[f_y:f_y + shape[0], f_x:f_x + shape[1]] += distance
        print(out.shape)
        # cv.imshow("out", out.astype(np.uint8))
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        if border:
            out[f_y-10, f_x - 100: f_x + shape[1] + 100] = 255
            out[f_y+10, f_x - 100: f_x + shape[1] + 100] = 255
            out[f_y - 100:f_y + shape[0] + 100, f_x-10] = 255
            out[f_y - 100:f_y + shape[0] + 100:, f_x+10] = 255
            if d_v is not None:
                out_d[f_y-10, f_x - 100: f_x + shape[1] + 100] = 1
                out_d[f_y+10, f_x - 100: f_x + shape[1] + 100] = 1
                out_d[f_y - 100:f_y + shape[0] + 100, f_x-10] = 1
                out_d[f_y - 100:f_y + shape[0] + 100:, f_x+10] = 1
    
    if d_v is None:
        return out.astype(np.uint8)
    else:
        out = out / out_d
        return out.astype(np.uint8)
    

def findSubAreaAndOffsetImage(image1, image2, overlap, repeatability, direction="x", offset=False):
    """
    finds the subarea of the images and randomly offsets if desired
    image1: the first image (leftmost or lower)
    image2: the second image(rightmost or over)
    overlap: tuple of with (overlap_h, overlap_v) in pixels
    repeatability: the repeatability desired. The random shift will be pm repeatability
    direction: the relative direction between the images "x" or "y"
    offset: bool to say if one wants offset or not.

    returns: sub_img1, sub_img2, offset
    sub_img1: the sub imgae from image one
    sub_img2: the sub image from image two
    offset: (y, x) offset from the offset. 
    
    
    """
    x, y = 0, 0
    shape = image1.shape
    if direction == "x":
        image1_x = [shape[1] - overlap[1], shape[1]]
        image1_y = [0, shape[0]]
        image2_x = [0, overlap[1]]
        image2_y = [0, shape[0]]
    else:
        if direction != "y":
            print(f"Direction {direction} is not implemented, setting to 'y'")
        image1_x = [0, shape[1]]
        image1_y = [0, shape[0] - overlap[0]]
        image2_x = [0, shape[1]]
        image2_y = [overlap[0], shape[0]]
    
    if offset:
        x = np.random.randint(2*repeatability) - repeatability
        y = np.random.randint(2*repeatability) - repeatability
        if direction == "x":
            if x > 0:
                image2_x[0] += x
                image2_x[1] += x
            else:
                image1_x[1] -= abs(x)
                image2_x[1] -= abs(x)
            if y > 0:
                image1_y[0] += y 
                image2_y[1] -= y 
            else:
                image1_y[1] -= abs(y)
                image2_y[0] += abs(y)
        else:
            if y > 0:
                image2_y[0] += y
                image2_y[1] += y
            else:
                image1_y[1] -= abs(y)
                image2_y[1] -= abs(y)
            if x > 0:
                image1_x[0] += x 
                image2_x[1] -= x 
            else:
                image1_x[1] -= abs(x)
                image2_x[0] += abs(x)

    
    sub_img1 = image1[image1_y[0]:image1_y[1], image1_x[0]:image1_x[1]]
    sub_img2 = image2[image2_y[0]:image2_y[1], image2_x[0]:image2_x[1]]
    return sub_img1, sub_img2, (y, x)
        


def offsetImagesRandom(image1, image2, repeatability, direction="x"):
    shape = image1.shape

    x = np.random.randint(2*repeatability) - repeatability
    y = np.random.randint(2*repeatability) - repeatability
    image1_x = [0, shape[1]]
    image1_y = [0, shape[0]]
    image2_x = [0, shape[1]]
    image2_y = [0, shape[0]]
    if direction == "y":
        if y < 0:
            image1_y[1] = shape[0] - abs(y)
        else:
            image2_y[0] = y
        if x < 0: 
            image1_x[0] = abs(x)
            image2_x[1] = shape[1] + x
        else:
            image1_x[1] = shape[1] - x
            image2_x[0] = x
    else:
        if direction != "x":
            print("direction is neither x or y, set to y")

        if x < 0:
            image1_x[1] = shape[0] - abs(x)
        else:
            image2_x[0] = x
        if y < 0: 
            image1_y[0] = abs(y)
            image2_y[1] = shape[0] + y
        else:
            image1_y[1] = shape[0] - y
            image2_y[0] = y



    img1 = image1[image1_y[0]:image1_y[1], image1_x[0]:image1_x[1]]
    img2 = image2[image2_y[0]:image2_y[1], image2_x[0]:image2_x[1]]
    print(img1.shape)
    print(img2.shape)


    print()
    print(f"OFFSET: {x, y} (x, y)")
    print()

    return img1, img2



