


from Images import Images
import numpy as np
import cv2
import m2stitch
import pandas as pd
from os import path
import time
import os
import re
from pathlib import Path


class ashlar(Images):

    
    def __init__(self, fileFolder: str, outputFolder: str, filenameFormat: str) -> None:
        super().__init__(fileFolder, outputFolder, filenameFormat)


    def searchInputFolder(self, folder: Path, filenameFormat: str) -> bool:
        
        firstFolder = True
        numFiles = len(os.listdir(folder))
        i = 0
        for file_path in os.listdir(folder):
            print(f"Parsing file {i} of {numFiles}")
            i += 1
            if os.path.isfile(os.path.join(folder, file_path)):
                s = re.search(filenameFormat, file_path)
                if s:
                    row = int(s.group(1))
                    #img = cv2.imread(folder + "\\" + file_path)
                    # if img is None:
                    #     continue
                    # if firstFolder:
                    #     self.shape = img.shape
                    # elif self.shape != img.shape:
                    #     print("Not all images are the same shape")
                    #     return False
                    # if img.shape[0] < min_res[0]:
                    #     min_res[0] = img.shape[0]
                    # if img.shape[1] < min_res[1]:
                    #     min_res[1] = img.shape[1]

                    # if row >=5:
                    #     continue 

                    self.imgs[file_path] = (row, 0)
                    firstFolder = False
            if firstFolder:
                return False


        return self.checkRowCols()

    def stitch(self):

        imgs = " ".join([self.fileFolder + "\\" + image for image in self.imgs.keys()])

        


        cmd = R"ashlar " + imgs + " --align-channel 0 -f ashlar_output_cycle{cycle}_channel{channel}.ome.tiff"
        print(cmd)
        start = time.time()
        returned_value = os.system(cmd)  # returns the exit code in unix
        end = time.time()
        self.timing["stitch"] = end - start
        print('returned value:', returned_value)

        # result_image_file_path = path.join(script_path, "stitched_image.npy")
        # np.save(result_image_file_path, stitched_image)

