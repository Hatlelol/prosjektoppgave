import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time


lowerLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_000.ome.tif"
upperLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_019.ome.tif"
upperRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_019.ome.tif"
lowerRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_000.ome.tif"
lowerLeft_n = R"C:\data\dsd_10\dsd_10\dsd_10_MMStack_1-Pos000_000.ome.tif"
upperLeft_n = R"C:\data\dsd_10\dsd_10\dsd_10_MMStack_1-Pos000_014.ome.tif"

scene_img = cv.imread(lowerLeft_n, cv.IMREAD_GRAYSCALE)
scene_img = scene_img*(255/np.amax(scene_img))
scene_img = scene_img.astype(np.uint8)
print(np.amax(scene_img))
print(np.amin(scene_img))

assert scene_img is not None, "file could not be read, check with os.path.exists()"
scene_img2 = scene_img.copy()


template = cv.imread(R"C:\data\grid\corner.jpg", cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

# cv.imshow("Corner", scene_img)
# cv.imshow("Template", template)

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED'] #, # 'cv.TM_CCORR',
 #'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
print(scene_img.shape)
print(template.shape)
for meth in methods:
    img = scene_img2.copy()
    method = eval(meth)
    # Apply template Matching
    s = time.time()
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    e = time.time()
    print(f"Time used for {meth} is {e - s} seconds")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()