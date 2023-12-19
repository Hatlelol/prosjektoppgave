import numpy as np
import cv2


gray = cv2.imread(R"C:\Users\sondr\Downloads\dsd_10\dsd_10\Out\img-stitched-0.jpg", cv2.IMREAD_GRAYSCALE)

height = 300
width = height * gray.shape[0]/gray.shape[1]
dim = (width, height) 


gray = gray[::2, ::2]
gray = gray[::2, ::2]
gray = gray[::2, ::2]
gray = gray[::2, ::2]
gray = gray[::2, ::2]



kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
smooth = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)


# divide gray by morphology image
division = cv2.divide(gray, smooth, scale=255)

# threshold
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 4)

#erode
erosion_shape = cv2.MORPH_RECT
erosion_size = 1
element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
print(element)
erode = cv2.erode(smooth, element)

blur = cv2.GaussianBlur(gray, (13,13), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Two pass dilate with horizontal and vertical kernel
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,5))
dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,9))
dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

# Find contours, filter using contour threshold area, and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for c in cnts:
    area = cv2.contourArea(c)
    if area > 200:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)

cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dilate)
cv2.imshow('image', image)
cv2.waitKey()
# show results
# cv2.imshow('smooth', smooth)  
# cv2.imshow('division', division)  
# cv2.imshow('thresh', thresh)  
# cv2.imshow("erode", erode)
cv2.imshow("erode - smooth", smooth - erode)
print(len(contours))
erode = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(erode, contours, -1, (125,255,0), 3)
cv2.imshow("erode2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
