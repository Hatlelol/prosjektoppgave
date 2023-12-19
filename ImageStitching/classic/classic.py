
from Images import Images
import numpy as np
import cv2

class Classic(Images):

    
    def __init__(self, fileFolder: str, outputFolder: str, filenameFormat: str) -> None:
        super().__init__(fileFolder, outputFolder, filenameFormat)

    def stitch(self):
        print((self.shape[0]*self.maxCol, self.shape[1]*self.maxRow))
        out = np.zeros((self.shape[0]*(self.maxCol + 1), self.shape[1]*(self.maxRow + 1), 3), dtype=np.uint8)

        for image, place in self.imgs.items():
            img = self.openImage(self.fileFolder + "\\" + image)

            out[img.shape[0]*place[0]:img.shape[0]*(place[0] + 1), img.shape[1]*place[1]: img.shape[1]*(place[1] + 1)] = img
        
        return out





