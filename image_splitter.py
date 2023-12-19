import numpy as np
import cv2
n = 8
while n % 2 == 0:
    maxPrime = 2
    n >>= 1     # equivalent to n /= 2

    print(n)


class ImageSplitter:


    _overlap = None #ovelap in percent
    _numImages = None
    _dimensions = None 
    _error = 0
    


    def __init__(self, image):

        self.full_image = cv2.imread(image) 
        self.full_dim = self.full_image.shape
        self.dtype = self.full_image.dtype

    def defineMaxAxis(self, a: int, b: int)-> (int, int):
        if self.shape[0] > self.shape[1]:
            return (max(a, b), min(a, b))
        
        return (min(a, b), max(a, b))


    def setNumImages(self):
        # Sets the value of NumImages from the self._dimensions variable
        self._numImages = self._dimensions[0] * self._dimensions[1]
    
    def setDimensions(self):
        # Sets self._dimensions based on the value of self._numImages. May change Num_images if it is not divideable on itself. 
        if np.sqrt(self._numImages).is_integer():
            self._dimensions = (int(np.sqrt(self._numImages)), int(np.sqrt(self._numImages)))
            return
        k = 0
        while n % 2 == 0:
            maxPrime = 2
            n >>= 1     # equivalent to n /= 2
            k += 1
        
        if n == 1:
            if i == 0:
                self._dimensions = self.defineMaxAxis(1, self._numImages)
                return

            self._dimensions = self.defineMaxAxis(k*2, int(self._numImages/(k*2)))
            return

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                maxPrime = i
                n = n /i
        self._dimensions = self.defineMaxAxis(maxPrime, int(self._numImages/maxPrime))




    def getDivision(self, length: int, slices: int, overlap:int, error: float =None) -> (int, int):
        increment = int(length/(slices + overlap/100))
        if error is not None:
            max_err_inc = slices*increment*error/100
            length -= max_err_inc
            increment = int/length/(slices + overlap/100)

        wHalf = int((increment + increment*overlap/100)/2)
        inc = increment
        for i in range(10):
            mid = wHalf + i*increment
            print(f"mid: {mid}, l: {mid - wHalf} r: {mid + wHalf} w: {mid + wHalf - (mid - wHalf)}")

    return wHalf, increment

        

    def split(self):
        if self._dimensions is None:
            if self._numImages is None:
                self._dimensions = (2, 2)
                self.setNumImages()            
            self.setDimensions()
        if self._numImages is None:
            self.setNumImages()

        if self._overlap is None:
            self._overlap = 10
        
        (x_wHalf, x_increment) = self.getDivision(self._dimensions[0], self.shape[0])
        (y_wHalf, y_increment) = self.getDivision(self._dimensions[1], self.shape[1])

        imgs = np.zeros((x_wHalf*2, y_wHalf*2, self._numImages), dtype=int)


        for i in range(self.shape[0]):
            x_mid = x_wHalf + i*x_increment
            for j in range(self.shape[1]):
                y_mid = y_wHalf + j*y_increment
                imgs[i*self.shape[0] + j] = self.full_image[x_mid - x_wHalf: x_mid+x_wHalf, y_mid - y_wHalf: y_mid + y_wHalf]
