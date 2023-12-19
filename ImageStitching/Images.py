import os
from pathlib import Path
import re
import cv2
import time


class Images():

    imgs = {}
    shape = (0, 0)

    timing = {
        "full": 0,
        "stitch": 0,
        "images": [],
        "prep": 0
    }


    def openImage(self, img):
        print(img)
        out = cv2.imread(img) 
        return out


    def checkRowCols(self):
        vals = self.imgs.values()


        max_row = max(row for row, _ in vals)
        max_col = max(col for _, col in vals)
        expected_numbers = {(row, col) for row in range(max_row + 1) for col in range(max_col + 1)}
        
        existing_numbers = set(vals)
        
        duplicates = set([item for item in vals if list(vals).count(item) > 1])
        missing_numbers = expected_numbers - existing_numbers
        
        

        if missing_numbers or duplicates:
            return False
        
        self.maxRow = max_row
        self.maxCol = max_col
        return True
            



    def searchInputFolder(self, folder: Path, filenameFormat: str) -> bool:
        
        if re.search(r"[\w.,\-()]*r{r+}_c{c+}.+\.[a-z]{3}", filenameFormat) is not None:
            self.filenamePatternType = "ROWCOW"
        
        elif re.search(r"[\w.,\-()]+pos{p+}[\w.,\-()]*\.[a-z]{3}", filenameFormat) is not None:
            self.filenamePatternType = "SEQUENTIAL"

        firstFolder = False
        numFiles = len(os.listdir(folder))
        i = 0
        for file_path in os.listdir(folder):
            print(f"Parsing file {i} of {numFiles}")
            i += 1
            if os.path.isfile(os.path.join(folder, file_path)):
                s = re.search(filenameFormat, file_path)
                # print(filenameFormat)
                # print(file_path)
                if s:
                    # print(s)    
                    row = int(s.group(1))
                    col = int(s.group(2))
                    # img = cv2.imread(os.path.join(folder, file_path), flags=cv2.IMREAD_GRAYSCALE)
                    # cv2.imshow(f"r{row},c{col}", img)
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

                    if row >=2 or col >= 2:
                        continue 

                    self.imgs[file_path] = (row, col)
                    firstFolder = False
            if firstFolder:
                return False


        return self.checkRowCols()

    def __init__(self, fileFolder: str, outputFolder: str, filenameFormat: str) -> None:
        if not Path(fileFolder).exists():
            if not (Path.cwd() / fileFolder).exists():
                print(f"Could not find the folder input: {fileFolder}")
                return
            fileFolder = (Path.cwd() / fileFolder)
        
        if not Path(outputFolder).exists():
            if not (Path.cwd() / outputFolder).exists():
                print(f"Could not find the output folder: {outputFolder}")
                return
            outputFolder = (Path.cwd() / outputFolder)

    
        self.fileFolder = fileFolder
        self.outputFolder = outputFolder
        self.filenameFormat = filenameFormat
        start = time.time()
        if self.searchInputFolder(fileFolder, filenameFormat):
            end = time.time()
            self.timing["prep"] = end - start

        
    def printTime(self):
        print("----- TIMING ------")
        print()
        temp = self.timing['stitch'] + self.timing['prep']
        print(f"FULL TIME: {self.timing['full'] if self.timing['full'] > temp else temp}")
        print(f"PREP TIME: {self.timing['prep']}")
        if self.timing['images']:
            print(f"IMAGES: {' '.join(['img' + i + ': ' + t for i, t in enumerate(self.timing['images'])])}")

        print(f"STITCH TIME: {self.timing['stitch']}")
