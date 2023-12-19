import numpy as np
import cv2 as cv
import sys  
sys.path.insert(0, R'C:\Users\sondr\OneDrive - NTNU\KodeTesting\FinishedClass\Python')
from image import Image
from helpFunctions import Position


path1 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos000_001.ome.jpg"
path2 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos001_001.ome.jpg"
path3 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos002_001.ome.jpg"
path4 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos003_001.ome.jpg"
path5 = R"C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos004_001.ome.jpg"
paths = [path1, path2, path3, path4, path5]


img_Test = cv.imread(path1, cv.IMREAD_GRAYSCALE)
max_error = 1
overlap_percent = 10 + max_error
vertical_overlap = int(img_Test.shape[1]*overlap_percent/100)
horizontal_overlap = int(img_Test.shape[0]*overlap_percent/100)
col = 1
images = []
for path in paths:
    expected_pos = Position(col*img_Test.shape[1], 0)
    col += 1
    images.append(Image(path=path, vertical_overlap=vertical_overlap, horizontal_overlap=horizontal_overlap, expected_pos=expected_pos, shape=img_Test.shape))


for image_1, image_2 in zip(images[:-1], images[1:]):
    print(image_1.path)
    print(image_2.path)

    image_2.stitch(image_1)