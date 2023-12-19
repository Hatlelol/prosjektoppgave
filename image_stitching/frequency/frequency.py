import numpy as np
from  wrapper.wrapperclas import stitch
import time

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


class Frequency(stitch):
    # TODO: make functions for finding next peaks


    _peak_bias = 2.5


    def posToIndex(self, position, shape):
        return np.where(position < 0, -position, shape - position)


    def indexToPos(self, index, shape):
        shape = np.array(shape)
        
        return np.where(index - (shape/2).astype(int) > 0, shape - index, -index)

    def generatePeakBias(self, shape: np.ndarray, position: np.ndarray):
        f = self._peak_bias # TODO: optimize this constant, distance penalty factor
        
        id = self.posToIndex(position, shape)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        distances = np.sqrt((x - id[1])**2 + (y - id[0])**2)

        distances = np.exp(-f*(distances/np.amax(distances))**2)
        return distances
    
    def setPeakBias(self, val):
        self._peak_bias = val

    def setPeakBiasDistance(self, repeatability):
        shape = self.shape
        id = self.posToIndex(self.position, shape)
        s = np.sqrt(np.sum((shape - id)**2))
        self._peak_bias = 0.1*(s/repeatability) #0.1 comes from ln(0.9), the position where peak bias funciton is 0.9


    def __init__(self, image1: np.ndarray, image2: np.ndarray, position: np.ndarray = None, repeatability: int = None, method = "NARROW") -> None:
        super().__init__(image1, image2)
        if position is None:
            self.position = (0, 0)
            self.repeatability = None
        else:
            self.position = position
            self.repeatability = repeatability
        self.method = method if method != None else "NARROW"

    def stitch(self, bias=True):
        """
        Needs to do the stitching and set the best result and timing
        
        """

        start_time = time.time()
        correlation = phaseCorrelation(self.image1, self.image2)
        correlation_end_time = time.time()
        
        if bias:
            peak_bias = self.generatePeakBias(correlation.shape, self.position)
            correlation = correlation*peak_bias
        
        peak = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        result = self.indexToPos(np.array(peak), correlation.shape)

        end_time = time.time()
        self.correlation = correlation

        self._setTime(end_time - start_time)
        self._setResult(result)

        self._has_stitched = True

        self.setTimeSummary(f"Total time use: {end_time - start_time}\nCorrelation time use: {correlation_end_time - start_time}\nBias time use: {end_time - correlation_end_time}\n")

        print(f"Correlation shape is equal to self shape : {correlation.shape == self.shape}")
