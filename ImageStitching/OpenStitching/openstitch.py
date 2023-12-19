from Images import Images
import numpy as np
import cv2
from stitching import Stitcher
from stitching.seam_finder import SeamFinder
import time


class OpenStitch(Images):

    
    def __init__(self, fileFolder: str, outputFolder: str, filenameFormat: str) -> None:
        super().__init__(fileFolder, outputFolder, filenameFormat)

    def stitch(self):
        start = time.time()        
        settings = {"detector": "sift", "confidence_threshold": 0.2}
        stitcher = Stitcher(**settings)

        panorama = stitcher.stitch([self.fileFolder + "\\" + image for image in self.imgs.keys()])
        end = time.time()
        self.timing["stitch"] = end - start

        return panorama

