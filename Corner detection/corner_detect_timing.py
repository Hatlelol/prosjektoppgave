import numpy as np
import cv2 as cv
import time
from verification.generateTest import generateAndDrawRectangles
from filters.hough_transform import *



def runIterationsHarris(N, debug=False):
    """
    Runs iteration over harris corner detector. Generates two rectangles with generate and draw rectangles on a 1000x1000 gray image. 

    N: number of iterations
    
    returns
    start_times: N long array of all the start times
    sub_times: N long array of all the sub times, i,e only time of the corner detector
    sub_times_2: N long array of the time it takes for the corner locations to be detected
    end_times: N long array of the time it takes until the corners are drawn on the input image. 
    score_array: N long array of ints, If 0 the corners were guessed correctly, if otherwise no. Higher for more incorrect
    """
    start_times = np.full(N, 0, float)
    sub_times   = np.full(N, 0, float)
    sub_times_2   = np.full(N, 0, float)
    end_times   = np.full(N, 0, float)
    score_arr   = np.full(N, 0, np.uint8)

    img_src = np.full((1000, 1000), 126, np.uint8)
    
    if N > 1000:
        print_interval = int(N/100)
    elif N > 100:
        print_interval = int(N/10)
    else: 
        print_interval = 1
    print("starting iterations")
    for k in range(N):

        #from https://answers.opencv.org/question/186538/to-find-the-coordinates-of-corners-detected-by-harris-corner-detection/
        if k % print_interval == 0:
            print(f"iteration {k} of {N}")
        img = img_src.copy()
        img, rectangle1, rectangle2 = generateAndDrawRectangles(img, 1.5, 0, 2, return_rectangles=True)
        # img = img[::8, ::8]
        num = 0.2
        gray = np.float32(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        out = img.copy()
        
        start = time.time()
        dst = cv.cornerHarris(gray,50,29, num)
        sub_time = time.time()

        _, dst = cv.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        sub_time_2 = time.time()
        
        out[dst>0.1*dst.max()] = [0, 0, 255]
        end_time = time.time()

        r_1 = [0, 0, 0, 0]
        r_2 = [0, 0, 0, 0]

        inaccurate_score = 0
        for corner in corners[1:]:
            lowest_corner_val = np.sum(np.abs(corner - rectangle1[0]))
            id_1 = 0
            for i in range(1, len(rectangle1)):
                if np.sum(np.abs(corner - rectangle1[i])) < lowest_corner_val:
                    lowest_corner_val = np.sum(np.abs(corner - rectangle1[i]))
                    id_1 = i
            id_2 = -1       
            for i in range(0, len(rectangle2)):
                if np.sum(np.abs(corner - rectangle2[i])) < lowest_corner_val:
                    lowest_corner_val = np.sum(np.abs(corner - rectangle2[i]))
                    id_2 = i
            if lowest_corner_val > 20:
                inaccurate_score += 1

            if id_2 == -1:
                r_1[id_1] += 1
            else:
                r_2[id_2] += 1

        score = np.sum(np.abs(np.concatenate((np.array(r_1), np.array(r_2)), axis=None) - 1))
        if score == 0:
            if inaccurate_score > 0:
                score = inaccurate_score + 20

        score_arr[k] = score
        start_times[k] = start
        sub_times[k] = sub_time
        sub_times_2[k] = sub_time_2
        end_times[k] = end_time
        if debug:
            print(corners[0])
            for i in range(1, len(corners)):
                print(corners[i])
                c = np.int32(corners[i])[::-1]
                out[c[0]-2:c[0]+2, c[1]-2:c[1]+2] = [255, 0, 0]


            #between 0.2 and 0.1 found best
            cv.imshow("dst", out)
            cv.waitKey(0)

    return start_times, sub_times, sub_times_2, end_times, score_arr

# print(np.array(end_times) - np.array(start_times))
# print(score_arr)


def runIterationsHough(N, debug=True):
    """
    Runs iteration over Hough line detector. Generates two rectangles with generate and draw rectangles on a 1000x1000 gray image. 

    N: number of iterations
    
    returns
    start_times: N long array of all the start times
    sub_times: N long array of all the sub times, i,e only time of generating the hough lines
    end_times: N long array of the time it takes until the two rectangles are found. 
    score_array: N long array of ints, If 0 the corners were guessed correctly, if otherwise no. Higher for more incorrect
    """
    start_times = np.full(N, 0, float)
    sub_times   = np.full(N, 0, float)
    end_times   = np.full(N, 0, float)
    score_arr   = np.full(N, 0, np.uint8)

    img_src = np.full((1000, 1000), 126, np.uint8)
    if N > 1000:
        print_interval = int(N/100)
    elif N > 100:
        print_interval = int(N/10)
    else: 
        print_interval = 1
    print("starting iterations")
    for k in range(N):

        #from https://answers.opencv.org/question/186538/to-find-the-coordinates-of-corners-detected-by-harris-corner-detection/
        if k % print_interval == 0:
            print(f"iteration {k} of {N}")
        img = img_src.copy()
        img, rectangle1, rectangle2 = generateAndDrawRectangles(img, 1.5, 0, thickness=1, return_rectangles=True)

        start = time.time()
        lines = baseTransform(img, single_line=True)
        if lines is None:
            score_arr[k] = 16
            start_times[k] = start
            sub_times[k] = time.time()
            end_times[k] = time.time()
            continue
        
        sub_time = time.time()
        inner_rec, outer_rec = findRectanglePairTiming(lines, 0, 10, img)
        end_time = time.time()


        if inner_rec is None or outer_rec is None:
            score = 15

        else:
            r_1 = [0, 0, 0, 0]
            r_2 = [0, 0, 0, 0]

            inaccurate_score = 0 

            for corner in inner_rec:
                lowest_corner_val = np.sum(np.abs(corner - rectangle1[0]))
                id_1 = 0
                for i in range(1, len(rectangle1)):
                    if np.sum(np.abs(corner - rectangle1[i])) < lowest_corner_val:
                        lowest_corner_val = np.sum(np.abs(corner - rectangle1[i]))
                        id_1 = i
                if lowest_corner_val > 20:
                    inaccurate_score += 1
                r_1[id_1] += 1
                
            for corner in outer_rec:
                lowest_corner_val = np.sum(np.abs(corner - rectangle2[0]))
                id_2 = 0
                for i in range(1, len(rectangle2)):
                    if np.sum(np.abs(corner - rectangle2[i])) < lowest_corner_val:
                        lowest_corner_val = np.sum(np.abs(corner - rectangle2[i]))
                        id_2 = i
                if lowest_corner_val > 20:
                    inaccurate_score += 1
                r_2[id_2] += 1


      
            
            score = np.sum(np.abs(np.concatenate((np.array(r_1), np.array(r_2)), axis=None) - 1)) + inaccurate_score
            if score == 0:
                if inaccurate_score > 0:
                    score = inaccurate_score + 20
        
        score_arr[k] = score
        start_times[k] = start
        sub_times[k] = sub_time
        end_times[k] = end_time
        

    return start_times, sub_times, end_times, score_arr
