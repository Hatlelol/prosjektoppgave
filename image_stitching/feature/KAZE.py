
import numpy as np
from wrapper.wrapperclas import stitch
import time
import cv2 as cv
from helpFunctions.helpFunctions import imshow
import time

class KAZE(stitch):


    def __init__(self, image1: np.ndarray, image2: np.ndarray,  position: np.ndarray = None, repeatability: int = None, method = "NARROW") -> None:
        """
        class to run kaze
        
        """

        super().__init__(image1, image2)
        if position is None:
            self.position = (0, 0)
            self.repeatability = None
        else:
            self.position = position
            self.repeatability = repeatability
        self.method = method if method != None else "NARROW"


    def stitch(self):
        
        #Can set upright or not to be faster
        
        start_time = time.time()
        # Create kaze object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        kaze = cv.KAZE_create()

    
    
        # Find keypoints and descriptors directly
        kp1, des1 = kaze.detectAndCompute(self.image1, None)
        end_kp1 = time.time()
        kp2, des2 = kaze.detectAndCompute(self.image2, None)
        end_kp2 = time.time()

        if des1 is None or des2 is None or len(des1) == 1 or len(des2) == 1:
            self.has_failed = True

            displacement = self.position
            end_time = end_kp2

            self._setTime(end_time - start_time)
            self._setResult(displacement)

            self._has_stitched = True

            self.setTimeSummary(f"Finding keypoints in image1: {end_kp1 - start_time}\n\
Finding keypoints in image2: {end_kp2 - end_kp1}\n\
Finding matches and displacement: \t{np.nan}\n\
Total time:\t\t{end_time - start_time}\n")

            self.setError("FoundNoKeypoints")
            return
        
        displacement, match_time = self.matchKeypoints(kp1, des1, kp2, des2)
       
        end_time = time.time()

        self._setTime(end_time - start_time)
        self._setResult(displacement)

        self._has_stitched = True

        self.setTimeSummary(f"Finding keypoints in image1: {end_kp1 - start_time}\n\
Finding keypoints in image2: {end_kp2 - end_kp1}\n\
Finding matches and displacement: \t{match_time}\n\
Total time:\t\t{end_time - start_time}\n")


    def foundSift(self):
        return self._found_sift
        
    


    def matchKeypoints(self, kp1, des1, kp2, des2):
        start_time = time.time()
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_LSH = 6
    
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good=[]
        for k in matches:
            if len(k) != 2:
                continue
            m = k[0]
            n = k[1]
            if m.distance < 0.7*n.distance:
                good.append(m)


        if len(good) > 1:

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            full = src_pts - dst_pts
            # print(full)
            displacement = -np.int32(np.round(np.median(full, axis= 0)))[0][::-1]
            end_time = time.time()
            if self.debug:

                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only inliers
                                flags = 2)
                
                matches_img = cv.drawMatches(self.image1,
                                                kp1,
                                                self.image2,
                                                kp2,
                                                good,
                                                None,
                                                **draw_params)
                imshow("Matches", matches_img)

            # displacement = self.checkRepeatability(displacement)

            self._found_sift = True
        else:
            self.has_failed = True
            displacement = self.position
            self._found_sift = False
            end_time = time.time()

        return displacement, end_time - start_time