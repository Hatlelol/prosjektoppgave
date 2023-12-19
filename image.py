import numpy as np
import cv2 as cv


def getOverlapPixels(shape, overlap, max_error=0, direction="RIGHT"):
    """
    Computes the number of pixels needed in the overlap by adding overlap and max_error and computing 
    the overlap based on the input shape and direction

    shape: touple of size two with the dimensions needed to compute the overlap
    overlap: the overlap in percent
    max_error: the max error in percent
    direction: the orientation of the overlap, in pracsis if shape[0] or shape[1] will be used

    return: (overlap+max_error)*shape[1/0] based on the direction
    
    """
    if direction == "RIGHT" or direction == "LEFT":
        return int(shape[1]*(overlap + max_error)/100)
    if direction == "UP" or direction == "DOWN":
        return int(shape[0]*(overlap + max_error)/100)
    print(f"the direction {direction} is not implemented")

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
        print("Image shapes needs to be equal")
        return

    G_1 = fft2d(image1)
    G_2 = fft2d(image2)
    c = G_1*np.conj(G_2)

    d = ifft2d(c/np.abs(c))

    return (np.abs(d)/np.amax(np.abs(d))).astype(image1.dtype)



def trimInput(img_1: np.ndarray, img_2: np.ndarray, overlap: int, direction="RIGHT"):
    """
    Returns only the overlapping regions of the input images img_1 and img_2
    
    img_1: Grayscale image 2d array
    img_2: Grayscale image 2d array
    overlap: maximum overlap between the images in pixels
    direction: the orientation of img_2 wrt img_1 

    returns: two arrays with either height or width scaled to be overlap + max_error percent of original    
    """
    if direction == "RIGHT":
        return img_1[:, int(img_1.shape[1] - overlap):], img_2[:, :overlap]
    if direction == "LEFT":
        return img_1[:, :overlap], img_2[:, int(img_2.shape[1]-overlap):]
    if direction == "UP":
        return img_1[:overlap, :], img_2[int(img_1.shape[0] - overlap):, :]
    if direction == "DOWN":
        return img_1[int(img_1.shape[0] - overlap):, :], img_2[:overlap, :]
    print(f"Direction {direction} is not implemented yet.")


def getDisplacement(correlation):
    vis_fig = np.where(correlation > 10E-10, correlation, 0)

    displacement = list(np.unravel_index(correlation.argmax(), correlation.shape))
    print(displacement)


    if displacement[0] > int(correlation.shape[0]/2):
        displacement[0] = -correlation.shape[0] + displacement[0]
    if displacement[1] > int(correlation.shape[1]/2):
        displacement[1] = -correlation.shape[1] + displacement[1]
    return displacement, vis_fig

class Image():


    path = None
    IMREAD_TYPE = cv.IMREAD_GRAYSCALE


    _dX = None
    _dY = None
    _overlap_region = {
        "top": None,
        "bott": None,
        "left": None,
        "right": None
    } 


    @property
    def dX(self):
        return self._dX
    
    @property
    def dY(self):
        return self._dY
    
    @property
    def overlap_region(self):
        return self._overlap_region



    debug = False

    def __init__(self, path: str, row, col) -> None:
        self.path = path
        self.row = row
        self.col = col



    def _stitch(self, path: str, overlap: int, max_error: int, direction: str) -> None:
        img1 = cv.imread(path, self.IMREAD_TYPE)
        img2 = cv.imread(self.path, self.IMREAD_TYPE)
        self.img = img2
        self.shape = img2.shape

        if img1.shape != img2.shape:
            print(f"Img1 and img2 does not have equal shapes {img1.shape}, {img2.shape}")
            return 

        overlap_p = getOverlapPixels(img1.shape, overlap, max_error, direction)

        img1_trim, img2_trim = trimInput(img1, img2, overlap_p, direction) 
        
        phase_corr = phaseCorrelation(img1_trim, img2_trim)

        displacement, fig = getDisplacement(phase_corr)

        if self.debug == True:
            # DO SOMETHING WITH FIG
            pass

        if direction == "RIGHT":
            self._computeDisplacementRight(displacement, overlap_p, img1.shape[0])
        elif direction == "LEFT":
            self._computeDisplacementLeft(displacement, overlap_p, img1.shape[0])
        elif direction == "UP":
            self._computeDisplacementUp(displacement, overlap_p, img1.shape[0])
        elif direction == "DOWN":
            self._computeDisplacementDown(displacement, overlap_p, img1.shape[0])
        


        # self._dX = displacement[0]
        # self._dY = displacement[1] - overlap_p
        # self._overlap_region["top"] = max(displacement[0], 0)
        # self._overlap_region["bott"] = img1.shape[0] - abs(displacement[0])
        # self._overlap_region["left"] = 0
        # self._overlap_region["right"] = overlap_p - displacement[1]


    def _computeDisplacementRight(self, displacement, overlap, shape):
        self._dX                      = displacement[0]
        self._dY                      = displacement[1] - overlap
        self._overlap_region["top"]   = max(displacement[0], 0)
        self._overlap_region["bott"]  = shape[0] - abs(displacement[0])
        self._overlap_region["left"]  = 0
        self._overlap_region["right"] = overlap - displacement[1] #TODO: maybe this is wrong, perhaps min(overlap, overlap - displacement[1])
    def _computeDisplacementLeft(self, displacement, overlap, shape):
        self._dX                      = displacement[0]
        self._dY                      = displacement[1] + overlap
        self._overlap_region["top"]   = max(displacement[0], 0)
        self._overlap_region["bott"]  = shape[0] - abs(displacement[0])
        self._overlap_region["left"]  = shape[1] - overlap - displacement[1]
        self._overlap_region["right"] = shape[1]
    def _computeDisplacementUp(self, displacement, overlap, shape):
        self._dX                      = displacement[0] + overlap
        self._dY                      = displacement[1]
        self._overlap_region["top"]   = shape[0] - overlap - displacement[1]
        self._overlap_region["bott"]  = shape[0]
        self._overlap_region["left"]  = max(displacement[1], 0)
        self._overlap_region["right"] = shape[1] - abs(displacement[1])
    def _computeDisplacementDown(self, displacement, overlap, shape):
        self._dX                      = displacement[0] - overlap
        self._dY                      = displacement[1]
        self._overlap_region["top"]   = 0
        self._overlap_region["bott"]  = overlap - displacement[0]
        self._overlap_region["left"]  = max(displacement[1], 0)
        self._overlap_region["right"] = shape[1] - abs(displacement[1])




    def stitchWithImg(self, path: str, overlap: int, max_error: int, direction: str = "RIGHT") -> None:

        if direction == "RIGHT":
            self._stitchRight(path, overlap, max_error)
        elif direction == "LEFT":
            self._stitchLeft(path, overlap, max_error)
        elif direction == "UP":
            self._stitchUp(path, overlap, max_error)
        elif direction == "DOWN":
            self._stitchDown(path, overlap, max_error)
        else:
            print(f"The direction {direction} is not implemented")


