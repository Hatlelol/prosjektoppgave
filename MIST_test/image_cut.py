import cv2
import urllib
import numpy as np

inp_img = "Trondheim.png"

numImages = 4
square = True
img = cv2.imread(inp_img, cv2.IMREAD_GRAYSCALE)

print(img.shape)

def checkPrime(num):
    if num > 1:
        for i in range(2, int(num/2)+1):
            if (num % i) == 0:
                return False
        else:
            return True
    else:
        return False


def getCropValues(numCrops, shape, overlap):
    max_dim = max(shape)
    min_dim = min(shape)
    ratio = min_dim/max_dim

    if square:
        disp = int((min_dim - overlap)/np.sqrt(numCrops))
        max_disp = min_disp = disp
        num_max = num_min = int(np.sqrt(numCrops))    
    elif np.sqrt(numCrops) % 1 == 0:
        max_disp = int((max_dim - overlap) / (np.sqrt(numCrops)))
        min_disp = int((min_dim - overlap*ratio) / (np.sqrt(numCrops)))
        num_max = num_min = int(np.sqrt(numCrops))
    elif checkPrime(numCrops):
        max_disp = int((max_dim - overlap) / numCrops)
        min_disp = 0
        num_max = numCrops
        num_min = 1
    else:
        num_min = int(np.floor(np.sqrt(numCrops)))
        num_max = int(numCrops / num_min)
        min_disp = int((min_dim - overlap*ratio) / num_min)
        max_disp = int((max_disp - overlap) / num_max)


    print(shape[1])
    print(max_dim)
    if max_dim == shape[1]:
        return min_disp, max_disp, num_min, num_max
    else:
        return max_disp, min_disp, num_max, num_min
    


def cropImage(img, numImage, overlap):

    y, x, y_n, x_n = getCropValues(numImage, img.shape, overlap)
    print(y, x, y_n, x_n)
    if max(img.shape) == img.shape[1]:
        x_overlap = overlap
        y_overlap = int(overlap * min(img.shape)/max(img.shape))
    else:
        x_overlap = int(overlap * min(img.shape)/max(img.shape))
        y_overlap = overlap


    x_it = y_it = 0
    for i in range(numImage):
        y_from = y*y_it
        y_to = y*(y_it + 1) + y_overlap
        x_from = x*x_it
        x_to = x*(x_it + 1) + x_overlap

        cropped_image = img[y_from:y_to,x_from:x_to]
        
        
        cv2.imshow('cropped', cropped_image)
        cv2.imwrite(f'img_r00{y_it}_c00{x_it}.jpg', cropped_image)
        x_it += 1
        if x_it == x_n:
            y_it += 1
            x_it = 0
        cv2.waitKey(0)


cropImage(img, 4, 200)
# cropped_image = img[80:250, 150:250]

# # display the cropped image
# cv2.imshow('cropped', cropped_image)

# # save the cropped image
# cv2.imwrite('Cropped_Test_Dog.jpg', cropped_image)

# # close viewing window

cv2.destroyAllWindows()

