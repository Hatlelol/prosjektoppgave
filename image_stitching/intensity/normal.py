
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

class Normal(stitch):

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

        

    def stitch(self):
        """
        
        
        
        """
        
        if self.method == "BROAD":
            x_id = np.arange(-2*self.repeatability**2, 2*self.repeatability**2 + 1, 1)
        elif self.method == "MEDIUM":
            x_id = np.arange(-2*self.repeatability, 2*self.repeatability + 1, 1)
        else:
            if self.method != "NARROW":
                print("Set mode to Narrow")
            x_id = np.arange(-self.repeatability, self.repeatability + 1, 1)
        

        
        y_id = x_id.copy()
        x_id = np.round(x_id -self.position[1]).astype(int)

        y_id = np.round(y_id - self.position[0]).astype(int)


        best_score = None
        best_score_pos = None

        sub_times = np.zeros((len(y_id), len(x_id)), float)
        start_time = time.time()
        result_arr = np.full((len(y_id), len(x_id)), 0, dtype=float)
        for i, y in enumerate(y_id):
            y_1_start, y_1_end, y_2_start, y_2_end = findPositions(self.shape[0], y)

            for j, x in enumerate(x_id):
                x_1_start, x_1_end, x_2_start, x_2_end = findPositions(self.shape[1], x)
                sub_time = time.time()

                score = np.sum(self.image1[y_1_start:y_1_end, x_1_start:x_1_end] - self.image2[y_2_start:y_2_end, x_2_start:x_2_end])
                
                sub_times[i, j] = time.time() - sub_time
                result_arr[i, j] = score
       
                if best_score is None:
                    best_score = score
                    best_score_pos = [y, x]
                else:
                    if score < best_score:
                        best_score = score
                        best_score_pos = [y, x]

        end_time = time.time()

        self._setTime(end_time - start_time)

        timesummary = f"Max time usage: \t Average time useage: \t Min time useage: \n"
        argmax = np.unravel_index(np.argmax(sub_times), sub_times.shape)
        argmin = np.unravel_index(np.argmin(sub_times), sub_times.shape)
        timesummary += f"({-y_id[argmax[0]]}, {-x_id[argmax[1]]}): {np.amax(sub_times.flatten()):.4f}\t {np.average(sub_times.flatten()):.4f}\t ({-y_id[argmin[0]]}, {-x_id[argmin[1]]}): {np.amin(sub_times.flatten()):.4f}\n"
        timesummary += f"\nTotal time: {end_time - start_time}\n"
        timesummary += f"Time used for whole array:\n"
        timesummary += f"\t {-x_id}"
        for i in range(len(y_id)):
            timesummary += f"{y_id[i]}: {result_arr[i]}\n"
 
        self.setTimeSummary(timesummary)

        self._setResult(-np.array(best_score_pos))      

        self._has_stitched = True



