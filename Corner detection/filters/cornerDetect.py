import numpy as np
import cv2 as cv
from tif import *
from hough_transform import *
import time

def filterImage(image):
    out_image = equalize(image, 200, 50, [100, 200])

    

    closed_mask = close(out_image)

    
    # c = np.where(closed_mask > 200)
    # max_tl = np.where(c[0] - c[1] == 0)[0][-1]
    # max_tr = np.where(np.amax(c[1]) - c[1] - c[0] == 0)[0][-1]

    # print(max_tl)
    # print(max_tr)   
    # print(min(c[1][max_tl], c[1][max_tr]))
    # out_image[:max(c[1][max_tl], c[1][max_tr]), :] = 0

    # out_image[:, :c[0][max_tl]] = 0
    # out_image[:, c[0][max_tr]:] = 0

    out_image[closed_mask > 200] = 0 

    out_image[out_image > 100] = 126

    # out_image = cv.Canny(out_image, 100, 200, None, 3, L2gradient=True)

    return out_image


# filename = R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg"
filename = R"C:\data\dsd_10\dsd_10\Out\img_stitch_bleng_vignettcorrection.png"
# filename = R"C:\data\grid\Fused_pic2_no_overlap_compute-1.tif"
# filename = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_000.ome.tif"

# img = read(filename, key=0)
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
print(np.amax(img))
img = (img*(255/np.amax(img))).astype(np.uint8)
# img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
# img = cv.imread(filename)
# gray = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
print(f"img: {img.shape}" )
print(np.amax(img))
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)[::8, ::8]
gray = img #[::8, ::8]
gray = cv.bilateralFilter(gray, d=9, sigmaColor=18, sigmaSpace=5)
cv.imshow("bilatFilter", gray)
cv.waitKey(0)
gray = filterImage(gray)
img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
print(f"gray: {gray.shape}" )
cv.imshow('dst1',gray)

gray = np.float32(gray)


# img = np.full((1000, 1000), 126, np.uint8)
# img = generateAndDrawRectangles(img, 1.5, 0, 2)
# gray = np.float32(img)
# img = np.cvtColor(img,)
# img = img[::8, ::8]
for num in np.arange(0.15, 0.25, 0.025):
    out = img.copy()
    start = time.time()
    dst = cv.cornerHarris(gray,50,29, num)
    sub_time = time.time()
    #between 0.2 and 0.1 found best
    out[dst>0.1*dst.max()] = [0, 0, 255]
    end_time = time.time()
    print(f"tot time {end_time - start}, sub_time: {sub_time-start}")
    cv.imshow("dst", out)
    cv.waitKey(0)

# for num in np.arange(0.1, 0.3, 0.05):
#     out = img.copy()
#     out[dst>num*dst.max()]=[0,0,255]
#     cv.imshow('dst',out)
#     cv.waitKey(0)
if cv.waitKey(0) & 0xff == 27:

    cv.destroyAllWindows()