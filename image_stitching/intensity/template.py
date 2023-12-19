
import numpy as np
import time
from wrapper.wrapperclas import stitch
import cv2 as cv
def imshow(text, image, destroy=False):
    cv.imshow(text, image)
    cv.waitKey(0)
    if destroy:
        cv.destroyAllWindows()

def findPositions(shape, pos):
    if pos < 0:
        return 0, shape - abs(pos), abs(pos), shape
    else:
        return abs(pos), shape, 0, shape - abs(pos)

class Template(stitch):

    # method = None
    # shape = None

    def __init__(self, image1: np.ndarray, image2: np.ndarray, position: np.ndarray, repeatability: int, method) -> None:
        if position is None:
            self.position = [0, 0]
        else:
            self.position = position
        print(repeatability)
        if repeatability is None:
            self.repeatability = int(min(image1.shape)/2)
        else:
            self.repeatability = repeatability
        # self.method = method
        super().__init__(image1, image2, method)

        

    def stitch(self, TEMPLATE_METHOD = 'cv.TM_CCOEFF_NORMED'):
        """
        
        
        
        """
        
        if self.position[0] > 0:
            img2 = self.image2[self.position[0]:, :]
        else:
            img2 = self.image2[:self.shape[0] + self.position[0], :]
        if self.position[1] > 0:
            img2 = self.image2[:, self.position[1]:]
        else:
            img2 = self.image2[:, :self.shape[1] + self.position[1]]

        img2 = img2[:int(img2.shape[0]/2), :int(img2.shape[1]/2)]



        meth = eval(TEMPLATE_METHOD)
        start_time = time.time()
        res = cv.matchTemplate(self.image1, img2, meth)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if TEMPLATE_METHOD in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        end_time = time.time()

        result = np.array(top_left)
        if self.position[0] > 0:
            result[0] -= self.position[0]
        
        if self.position[1] > 0:
            result[1] -= self.position[1]

        if self.debug:
            img = self.image1.copy()
            import matplotlib.pyplot as plt
            w, h = img2.shape[::-1]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img,top_left, bottom_right, 255, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()


        self._setResult(result)        

        self._setTime(end_time - start_time)

        self._has_stitched = True



