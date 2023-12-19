import sys
import math
import cv2 as cv
import numpy as np

import scipy.stats

def drawLines(image, lines):
    if lines is not None:
        print(lines)
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(image, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

    return image


def showImage(text, image):
    cv.imshow(text, image)
    cv.waitKey(0)


def resizeImage(image: np.ndarray, new_width: int) -> np.ndarray:
    """
    Resizes the input image with the desired new width
    """
    height, width = image.shape[:2]
    scaling_factor = new_width / width
    resized_image = cv.resize(image, (new_width, int(height * scaling_factor)))
    return resized_image


def equalize(image: np.ndarray, upper_bound: int, lower_bound: int, middle_bound: list or tuple or np.ndarray=None) -> np.ndarray:
    """
    Equalizes the image in three steps:
    Sets all image values over upper bound to 255
    Sets all iamge values under lower_bound to 0
    Sets all image values between middle_bound[0] and middle_bound[1] to 255/2
    If middle bound is None middle_bound[0] is lower_bound and middle_bound[1] is upper_bound

    image: np.ndarray in grayscale
    upper_bound: Values over this will be ceiled to 255
    lower_bound: values under this will be floored to 0
    middle bound: values between these will be set to 255/2

    return: filtered image
    """
    image[image > upper_bound] = 255
    image[image < lower_bound] = 0
    if middle_bound is None:
        image[np.logical_and(image > lower_bound, image < upper_bound)] = int(255/2)
    else:
        image[np.logical_and(image > middle_bound[0], image < middle_bound[1])] = int(255/2)

    return image

def close(image: np.ndarray, threshold: int=7*int(255/10), MORPH_TYPE: int =cv.MORPH_RECT) -> np.ndarray:
    """
    Does two closing operations in image for values above threshold. 
    
    image: input image
    threshold: values over threshold will be closed
    MORPH_TYPE: Type of kernel used, tested with cv.MORPH_RECT and cv.MORPH_DILATE

    return: closed image
    """
    mask = np.where(image > threshold, 254, 0).astype(np.uint8)

    kernel = cv.getStructuringElement(MORPH_TYPE, (21, 21))
    
    closed_image = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    closed_image = cv.morphologyEx(closed_image, cv.MORPH_CLOSE, kernel)
    return closed_image

def baseTransform(image: np.ndarray, debug: bool=False, single_line=False) -> np.ndarray:
    """
    Does preparation and the hough transform of the input image.
    The preparation is fine tuned to do detect rectangular chips with gray color with one dark rectangle inside.
    The surrounding of the chip is white with plenty of dark and black lines which are noise. 

    The preparation includes the steps: 
    Resize: resize to width of 800
    Equalize: floor and ceil gradients in the image
    Close: make the white regions close. 
    Canny edge detector
    Hough_lines
    
    return: An array with lines based in hough space. Shape (numLines, 1, 2) where lines[:, 0, 0] is rho and lines[:, 0, 1] is theta

    """

    


    equalized_image = equalize(image, 200, 50, [100, 200])

    if debug:
        showImage("Equalized image", equalized_image)

    closed_mask = close(equalized_image)

    if debug:
        showImage("Closing mask", closed_mask)

    equalized_image[closed_mask > 200] = 0 

    if debug:
        showImage("Closed image", equalized_image)

    
    canny_image = cv.Canny(equalized_image, 100, 200, None, 3, L2gradient=True)
    if single_line:
        img_blur = cv.GaussianBlur(canny_image, (5,5),sigmaX=10,sigmaY=10)
        if debug:
            showImage("blur", img_blur)
        canny_image = np.where(img_blur > int(np.amax(img_blur/2)), 255, 0).astype(np.uint8)

    if debug:
        showImage("Canny image", canny_image)

    lines = cv.HoughLines(canny_image, 1, np.pi/90, 150, None, 0, 0)


    
    if debug:
        if lines is not None:
            print(lines)
            color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(color_image, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

            showImage("Lines", color_image)

    return lines

def splitLines(lines: np.ndarray, centrum, timing=False):
    """
    Takes lines in hough space, These liens are sorted in increasing rho
    and grouped into two groups based on where rho has the highest increment.

    TODO: see what centrum does
    """
    # Takes an array of lienes in polar space and groups them based on where the rhos has the highest increment
    
    MIN_INCREMENT = 80

    rhos = lines[:, 0, 0]
    #Angles over pi/2 can have negative rhos, even if they are at approx same distance from center. Only if angle is around 0 and pi
    if np.any(lines[:, 0, 1] > 3*np.pi/4) and np.any(lines[:, 0, 1] < np.pi/4):
        rhos = np.where(lines[:, 0, 1] > np.pi/2, np.abs(rhos), rhos)

    sort_ids = np.argsort(rhos)
    if len(rhos) == 1:
        increment = rhos
    else:  
        increment = np.abs(rhos[sort_ids][1:]- rhos[sort_ids][:-1])


    if np.amax(increment) < MIN_INCREMENT:
        ids = np.array([np.where(rhos[sort_ids] > centrum[0])[0][0]])
    else:
        ids = np.where(increment > np.amax(increment)/2)[0] + 1
    
    regions = [[], []]
    if len(ids) > 2:
        for i in ids:
            if rhos[sort_ids][i] > centrum[0]:
                regions[0] = lines[sort_ids][:i]
                regions[1] = lines[sort_ids][i:]
                return regions
        if not timing:
            print("No large increment over centrum found")
    regions[0] = lines[sort_ids][:ids[0]]
    regions[1] = lines[sort_ids][ids[0]:]
    return regions

def getMaxAngleModeHalfPi(hist: np.ndarray, pm_resolution:int = 2) -> int:
    """
    Returns a value between 0 and 45 of representing the angle with the most lines, and most perpendicular lines.
    This is done to find the most prominent angles in the lines.
    Lines in hist are group with pluss minus pm_resolution of the hist

    hist: histogram of angles
    pm_resolution: plus minus grouping

    returns: the index in histogram where there are the most lines and perpendicular lines in total
    """

    hist_out = hist[0].copy()
    for i in range(1, pm_resolution + 1):
        hist_out += np.roll(hist[0], i)
        hist_out += np.roll(hist[0], -i)
    
    lower_half = hist_out[hist[1][:-1] < np.pi/2]
    upper_half = hist_out[hist[1][:-1] >= np.pi/2]

    return np.argmax(lower_half + upper_half)


def deleteRhoWithinResolution(lines: np.ndarray, resolution: int) ->np.ndarray:
    """
    Deletes lines where the rho is within pm resolution. 

    lines: np.ndarray
    resolution: pm rho resoltion. Everyting within rho - resolution < x < rho + resolution is deleted
    
    return: lines

    """

    #generated with help from chatgpt
    result = np.copy(lines)
    i = 0
    while i < len(result):
        # Find the indices within the resolution around the current value
        
        indices_to_delete = np.where(np.abs(result[:, 0, 0] - result[i, 0, 0]) <= resolution)[0]
        avg_rho = np.average(result[indices_to_delete][:, 0, 0])
        avg_theta = np.average(result[indices_to_delete][:, 0, 1])

        # Delete the indices within the resolution (excluding the current index)
        result = np.delete(result, indices_to_delete[1:], axis=0)
        result[indices_to_delete[0], 0, 0] = avg_rho
        result[indices_to_delete[0], 0, 1] = avg_theta
        i+=1

    return result

def getMaxPerpendicularLines(lines: np.ndarray, pm_degrees: int=2)-> (np.ndarray, np.ndarray):
    """
    Finds what lines with their perpendicular lines are most prominent

    lines: lines in hough space
    pm_Degrees: the number of neighbooring degrees to take into account

    returns:
    base_lines: the base lines
    perpendicular_lines: the lines perpendicular to base_lines    
    
    """

    degree_histogram = np.histogram(lines[:, 0, 1], np.linspace(0, np.pi, 91, endpoint=True))

    max_degrees_index = getMaxAngleModeHalfPi(degree_histogram, pm_degrees)

    #if max_degrees_index is under pm_egree we need to wraparound
    #Same at other end, accomadated for that:
    if max_degrees_index < pm_degrees:
        upper_bound_id = pm_degrees + max_degrees_index + 1 # less than this 
        lower_bound_id = len(degree_histogram[1]) - pm_degrees #more than this

        upper_bound_deg = degree_histogram[1][upper_bound_id]
        lower_bound_deg = degree_histogram[1][lower_bound_id]

        mask_degrees_base = np.logical_or(lines[:, 0, 1] < upper_bound_deg, lines[:, 0, 1] > lower_bound_deg)
        mask_degrees_perpendicular = np.logical_and(lines[:, 0, 1] < upper_bound_deg + np.pi/2, lines[:, 0, 1] > lower_bound_deg - np.pi/2)

    elif max_degrees_index > 45 - pm_degrees:
        # if argmax over 
        upper_bound_id = pm_degrees + max_degrees_index + 1 # less than this 
        lower_bound_id = max_degrees_index - pm_degrees #more than this

        upper_bound_deg = degree_histogram[1][upper_bound_id]
        lower_bound_deg = degree_histogram[1][lower_bound_id]

        mask_degrees_base = np.logical_and(lines[:, 0, 1] > lower_bound_deg, lines[:, 0, 1] < upper_bound_deg)
        mask_degrees_perpendicular = np.logical_or(lines[:, 0, 1] > lower_bound_deg + np.pi/2,\
                                                    lines[:, 0, 1] < upper_bound_deg - np.pi/2)
    else:

        #if argmax over pm_degrees
        upper_bound_id = pm_degrees + max_degrees_index + 1 # less than this 
        lower_bound_id = max_degrees_index - pm_degrees #more than this

        upper_bound_deg = degree_histogram[1][upper_bound_id]
        lower_bound_deg = degree_histogram[1][lower_bound_id]

        mask_degrees_base = np.logical_and(lines[:, 0, 1] >= lower_bound_deg, lines[:, 0, 1] <= upper_bound_deg)
        mask_degrees_perpendicular = np.logical_and(lines[:, 0, 1] >= lower_bound_deg + np.pi/2, lines[:, 0, 1] <= upper_bound_deg + np.pi/2)

    base_lines = lines[mask_degrees_base]
    perpendicular_lines = lines[mask_degrees_perpendicular]

    return base_lines, perpendicular_lines

def findSmallestParalell(region1:np.ndarray, region2:np.ndarray, timing=False) -> (int, int):
    """
    Finds the line with the larges rho in region 1 and smallest rho in region2 that are paralell within a resolution of pi/45
    
    returns:
    id_1: the index of the found line in region 1
    id_b: the index of the found line in region 2
    """
    id_1 = len(region1) - 1
    id_2 = 0
    inc_1 = True
    while np.abs(region1[id_1][0][1] - region2[id_2][0][1]) > np.pi/45:
        print(id_1, id_2)
        if id_1 < 0 or id_2 >= len(region2):
            if not timing:
                print("Could not find perfectly paralell lines")
            id_1 = len(region1) - 1
            id_2 = 0    
            break
        if inc_1:
            id_1 -= 1
            inc_1 = False
        else:
            id_2 += 1
            inc_1 = True
    return id_1, id_2




def hough_to_cartesian(rho, theta):
    #from chatgpt
    m = -1 / np.tan(theta) if np.abs(theta) > np.pi/4 else np.tan(theta)
    b = rho / np.sin(theta) if np.abs(theta) > np.pi/4 else rho / np.cos(theta)
    return m, b

def are_lines_intersecting(line1, line2, x_max, y_max):
    #from chatgpt

    x, y = intersectionFromPair(line1, line2) 

    if x is None or y is None:
        return False

    valid_intersections = (
        (0 <= x) & (x <= x_max) &
        (0 <= y) & (y <= y_max)
    )

    return np.any(valid_intersections)

def getNextSimilarAngle(baseline: float, lines:np.ndarray, shape: list or tuple or np.ndarray, rev:bool=False, timing = False) -> np.ndarray:
    """
    Finds the next largest or smaller line in lines with the same angle within pm pi/45 degrees
    
    angle: the angle of the next angle
    lines: the lines to search through, sorted by rho
    rev: bool to tell if the search should go from smalles to largest (False) or from largest to smallest (True)

    returns: one line
    """
    if rev:
        i = -1
        while abs(i) != len(lines) + 1:
            if not are_lines_intersecting(baseline, lines[i][0], shape[1], shape[0]):
                if np.abs(lines[i][0][1] - baseline[1]) < np.pi/45:
                    return lines[i]
            i -= 1 
        if not timing:
            print("could not find paralell line")
        return lines[-1]
    else:
        i = 0
        for i in range(len(lines)):
            
            if not are_lines_intersecting(baseline, lines[i][0], shape[1], shape[0]):
                if np.abs(lines[i][0][1] - baseline[1]) < np.pi/45:
                    return lines[i]
        if not timing:
            print("could not find paralell")
        return lines[0]     



def intersection(L1, L2):
        """
        finds the intersection between lines L1 and L2 where y=L[0]*x+L[1]

        input lines: [a, b] where y = a*x + b
        return point [y, x]        
        """
        D  = -L1[0]  + L2[0]
        Dx = L1[1]  - L2[1]
        Dy = -L1[0] * L2[1] + L1[1] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return y, x
        else:
            return None, None
    

def intersectionFromPair(l_1: np.ndarray, l_2: np.ndarray) -> np.ndarray:
    """
    Finds the intersection between two lines
    
    input lines: [rho, theta]
    output pount: [y, x]

    """
    if math.sin(l_1[1]) == 0:
        if math.sin(l_2[1]) == 0:
            return None, None
        x = -l_1[0] if l_1[1] > 1 else l_1[0] 
        y = l_2[0]
        return y, x
    if math.sin(l_2[1]) == 0:
        x = -l_2[0] if l_2[1] > 1 else l_2[0] 
        y = l_1[0]
        return y, x

    func_1 = (-math.cos(l_1[1])/math.sin(l_1[1]), l_1[0]/math.sin(l_1[1])) #y = l_1[0]*x+l_1[1]
    func_2 = (-math.cos(l_2[1])/math.sin(l_2[1]), l_2[0]/math.sin(l_2[1])) #y = l_1[0]*x+l_1[1]
    return intersection(func_1, func_2)

def pointsFromLinesRectangle(paralell1, paralell2):
    """
    Translates two sets of two paralell lines into points to draw a rectangle
    
    """
    points = np.array([
                       intersectionFromPair(paralell1[0], paralell2[0])[::-1], 
                       intersectionFromPair(paralell1[0], paralell2[1])[::-1], 
                       intersectionFromPair(paralell1[1], paralell2[1])[::-1],
                       intersectionFromPair(paralell1[1], paralell2[0])[::-1]
                       ])

    return points


def findRectanglePair(lines, resolution_degrees=2, resolution_rho=10, image=None):
    """
    Finds the smallest rectangle with middle approx in center and no lines running trough it
    Subsequently it finds the next larger rectangle from this line

    lines: The input lines on format where [:, 0, 0] is Rho and [:, 0, 1] is theta

    
    """

    base, perpendicular = getMaxPerpendicularLines(lines, resolution_degrees)


    base = deleteRhoWithinResolution(base, resolution_rho)
    most_prominant_angle, _ = scipy.stats.mode(base)
    base_regions = splitLines(base, most_prominant_angle[0])
    
    allLines = drawLines(image.copy(), base)
    allLines = drawLines(allLines, perpendicular)
    showImage("all lines", allLines)


    perpendicular = deleteRhoWithinResolution(perpendicular, resolution_rho)
    most_prominant_angle, _ = scipy.stats.mode(perpendicular)
    perpendicular_regions = splitLines(perpendicular, most_prominant_angle[0])
    print()
    print("BASE REGION")
    print(base_regions)
    print("PERPENDICULAR REGION")
    print(perpendicular_regions)
    print()

    region1 = drawLines(image.copy(), base_regions[0])
    region2 = drawLines(image.copy(), base_regions[1])
    region3 = drawLines(image.copy(), perpendicular_regions[0])
    region4 = drawLines(image.copy(), perpendicular_regions[1])

    cv.imshow("Base region 1", region1)
    cv.imshow("Base region 2", region2)
    cv.imshow("perpendicular region 1", region3)
    showImage("perpendicular region 2", region4)


    if len(base_regions) <= 1:
        print("could not form regions in base")
        return None
    if len(perpendicular_regions) <= 1:
        print("could not form regions in perpendicular")
        return None
    


    
    baseid0, baseid1 = findSmallestParalell(base_regions[0], base_regions[1])
    if baseid0 < 0 or baseid1 >= len(base_regions[1]):
        print("Could not find two rectangles")
        return None
    pair1 = [base_regions[0][baseid0][0], base_regions[1][baseid1][0]]

    perpid0, perpid1 = findSmallestParalell(perpendicular_regions[0], perpendicular_regions[1])
    if perpid0 < 0 or perpid1 >= len(perpendicular_regions[1]):
        print("Could not find two rectangles")
        return None
    pair2 = [perpendicular_regions[0][perpid0][0], perpendicular_regions[1][perpid1][0]]

    smallest_rectangle = pointsFromLinesRectangle(pair1, pair2)

    if len(base_regions[0][:baseid0]) == 0:
        print("Did not find enough base region lines")
        a = base_regions[0][baseid0][0]
    else:
        a = getNextSimilarAngle(pair1[0], base_regions[0][:baseid0], image.shape, rev=True)[0]

    if len(base_regions[1][baseid1 + 1:]) == 0:
        print("Did not find enough base region lines")
        b = base_regions[1][baseid1][0]
    else:
        b = getNextSimilarAngle(pair1[1], base_regions[1][baseid1 + 1:], image.shape, rev=False)[0]
        
    if len(perpendicular_regions[0][:perpid0]) == 0:
        print("Did not find enough perpendicular region lines")
        c = perpendicular_regions[0][perpid0][0]
    else:
        c = getNextSimilarAngle(pair2[0], perpendicular_regions[0][:perpid0], image.shape, rev=True)[0]

    if len(perpendicular_regions[1][perpid1 + 1:]) == 0:
        print("Did not find enough perpendicular region lines")
        d = perpendicular_regions[1][perpid1][0]
    else:
        d = getNextSimilarAngle(pair2[1], perpendicular_regions[1][perpid1 + 1:], image.shape, rev=False)[0]

    print("rectangle lines:")
    print(pair1[0], a)
    print(pair1[1], b)
    print(pair2[0], c)
    print(pair2[1], d)
    print("")

    larger_rectangle = pointsFromLinesRectangle([a, b], [c, d])

    return smallest_rectangle, larger_rectangle


def findRectanglePairTiming(lines, resolution_degrees=2, resolution_rho=10, image=None):
    """
    Finds the smallest rectangle with middle approx in center and no lines running trough it
    Subsequently it finds the next larger rectangle from this line

    lines: The input lines on format where [:, 0, 0] is Rho and [:, 0, 1] is theta

    
    """

    base, perpendicular = getMaxPerpendicularLines(lines, resolution_degrees)


    base = deleteRhoWithinResolution(base, resolution_rho)
    most_prominant_angle, _ = scipy.stats.mode(base)
    base_regions = splitLines(base, most_prominant_angle[0], timing=True)
    
 
    perpendicular = deleteRhoWithinResolution(perpendicular, resolution_rho)
    most_prominant_angle, _ = scipy.stats.mode(perpendicular)
    perpendicular_regions = splitLines(perpendicular, most_prominant_angle[0])


    if len(base_regions) <= 1:
        return None, None
    if len(perpendicular_regions) <= 1:
        return None, None
    


    
    baseid0, baseid1 = findSmallestParalell(base_regions[0], base_regions[1], timing=True)
    if baseid0 < 0 or baseid1 >= len(base_regions[1]):
        return None, None
    pair1 = [base_regions[0][baseid0][0], base_regions[1][baseid1][0]]

    perpid0, perpid1 = findSmallestParalell(perpendicular_regions[0], perpendicular_regions[1],  timing=True)
    if perpid0 < 0 or perpid1 >= len(perpendicular_regions[1]):
        return None, None
    pair2 = [perpendicular_regions[0][perpid0][0], perpendicular_regions[1][perpid1][0]]

    smallest_rectangle = pointsFromLinesRectangle(pair1, pair2)

    if np.any(smallest_rectangle == np.array([None, None])):
        return None, None

    if len(base_regions[0][:baseid0]) == 0:
        return None, None
    else:
        a = getNextSimilarAngle(pair1[0], base_regions[0][:baseid0], image.shape, rev=True, timing=True)[0]

    if len(base_regions[1][baseid1 + 1:]) == 0:
        return None, None
    else:
        b = getNextSimilarAngle(pair1[1], base_regions[1][baseid1 + 1:], image.shape, rev=False, timing=True)[0]
        
    if len(perpendicular_regions[0][:perpid0]) == 0:
        return None, None
    else:
        c = getNextSimilarAngle(pair2[0], perpendicular_regions[0][:perpid0], image.shape, rev=True, timing=True)[0]

    if len(perpendicular_regions[1][perpid1 + 1:]) == 0:
        return None, None
    else:
        d = getNextSimilarAngle(pair2[1], perpendicular_regions[1][perpid1 + 1:], image.shape, rev=False, timing=True)[0]


    larger_rectangle = pointsFromLinesRectangle([a, b], [c, d])

    if np.any(larger_rectangle == np.array([None, None])):
        return None, None

    return smallest_rectangle, larger_rectangle


def doTransform(image, canny_threshold1=50, canny_threshold2=200):
   

    # Loads an image
    # src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)[::4, ::4]
    # Check if image is loaded fine
    # if src is None:
    #     print ('Error opening image!')
    #     return -1

    dst = cv.Canny(image, canny_threshold1, canny_threshold2, None, 3, L2gradient=True)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 10)

    a = []
    b = []

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            p1 = [0, 0]
            p2 = [0, 0]
            if l[1] > l[3]:
                p1 = [l[2], l[3]]
                p2 = [l[0], l[1]]
            else:
                p1 = [l[0], l[1]]
                p2 = [l[2], l[3]]
            
            a_l = (p2[0] - p1[0])/(p2[1] - p1[1])
            b_l = p1[0] - a*p1[1]
            a.append(a_l)
            b.append(b_l)


            # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    # a = np.array(a)
    # b = np.array(b)

    

    return dst, cdst, cdstP
    # cv.namedWindow("Source", cv.WINDOW_NORMAL)
    # cv.imshow("Source", image)
    # cv.namedWindow("Canny", cv.WINDOW_NORMAL)
    # cv.imshow("Canny", dst)
    # cv.namedWindow("Detected Lines (in red) - Standard Hough Line Transform", cv.WINDOW_NORMAL)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.namedWindow("Detected Lines (in red) - Probabilistic Line Transform", cv.WINDOW_NORMAL)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    # cv.waitKey()

# doTransform(R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg")

# cv.destroyAllWindows()