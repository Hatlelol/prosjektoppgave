import numpy as np
import cv2 as cv
from verification.generateTest import *
from filters.hough_transform import *

# Example usage:
N = 1000
M = 1000

image_base = np.full((N, M, 3), (126, 125, 126), dtype=np.uint8)
for i in range(20):
    image = image_base.copy() 

    image = generateAndDrawRectangles(image, thickness=1)

    image[500:700, 500:700, :] = [126, 126, 126]
    cv.imshow("Random Rectangle", image)


    cv.waitKey(0)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resized_image = resizeImage(gray_image, 1000)

    lines = baseTransform(resized_image, True)
    if lines is None:
        print("Did not find any lines")
        continue
    
    img = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)

    img = drawLines(img, lines)
    cv.imshow("Lines", img)
    cv.waitKey(0)

    inner_rec, outer_rec = findRectanglePair(lines, 0, image=img)

    out_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)

    # out_image = drawRectangle(out_image, inner_rec, (0, 0, 255), 1)
    # cv.imshow("Inner rectangle", out_image)
    # cv.waitKey(0)
    # out_image = drawRectangle(out_image, outer_rec, (0, 0, 255), 1)
    # cv.imshow("Both rectangler", out_image)
    # cv.waitKey(0)


    cv.destroyAllWindows()