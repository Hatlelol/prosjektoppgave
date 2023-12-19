from classic.classic import Classic 
from OpenStitching.openstitch import OpenStitch
from MIST_py.m2stitch import MIST
from ASHLAR.ashlar import ashlar
import cv2
import numpy as np
inp = R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\ImageStitching\test_images"
realTestData = R"C:\data\MIST-Phase-Contrast-55x55"
output = R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\ImageStitching\output"

# inp = realTestData
inp = R"C:\Users\sondr\Downloads\dsd_10\dsd_10"
text = r"dsd_10_MMStack_1-Pos(\d{3})_(\d{3}).ome.tif"
text_trondheim = r"img_r(\d{3})_c(\d{3}).jpg"

c = MIST(inp, output, text)
img1 = c.stitch()
# cv2.imshow("img", c.stitch())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread(inp + "\\" + "img_r000_c001.jpg", flags=cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(inp + "\\" + "img_r000_c000.jpg", flags=cv2.IMREAD_GRAYSCALE)
# imgtot = np.stack((img, img2))
# print(imgtot.shape)

c.printTime()
cv2.imshow("stitch1", img1)
cv2.waitKey(0)

print("Starting openstitch")
k = OpenStitch(inp, output, r"img_r(\d{3})_c(\d{3}).jpg")
img2 = k.stitch()
k.printTime()
cv2.imshow("stitch2", img2)
cv2.waitKey(0)


if len(img1.shape) != len(img2.shape):
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def padArrays(arr1: np.ndarray, arr2: np.ndarray) -> (np.ndarray, np.ndarray):

    shape1 = arr1.shape
    shape2 = arr2.shape

    if shape1[0] > shape2[0]:
        arr2 = np.pad(arr2, [(0, shape1[0] - shape2[0]), (0, 0)])
    elif shape1[0] < shape2[0]:
        arr1 = np.pad(arr1, [(0, -shape1[0] + shape2[0]), (0, 0)])



    if shape1[1] > shape2[1]:
        arr2 = np.pad(arr2, [(0, 0), (0, shape1[1] - shape2[1])])
    elif shape1[1] < shape2[1]:
        arr1 = np.pad(arr1, [(0, 0), (0,  -shape1[1] + shape2[1])])

    return arr1, arr2

if img1.shape != img2.shape:
    img1, img2 = padArrays(img1, img2)


img_comb = img1 - img2
cv2.imshow("combined", img_comb)
cv2.waitKey(0)


cv2.destroyAllWindows()


