
import cv2 as cv
from helpFunctions import *
import sys


import time


# image = cv.imread(R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\ImageStitching\test_images\img_r000_c001.jpg", cv.IMREAD_GRAYSCALE)
# image = cv.imread(R"C:\data\space\SouthernRing\SouthernRingNebula.tif", cv.IMREAD_GRAYSCALE)[::2, ::2]
image = cv.imread(R"C:\data\space\ComsmicCliffs\cosmicCliff.tif", cv.IMREAD_GRAYSCALE)
print(np.amax(image), np.amin(image))

image = (image * (255/np.amax(image))).astype(np.uint8)



print(image.shape)
overlap = (np.array(image.shape)/6)*0.1
print(overlap)
tiles, c, c_r = splitImage(image, int(overlap[0]), int(overlap[1]), 6, 6, 0, 0)

# for i in range(3):
#     for j in range(3):
#         cv.imshow(f"{i}, {j}", tiles[i, j])
#         print(f"{c[i, j]}, {c_r[i, j]}")
#         cv.waitKey(0)


def time_stitch(img_a, img_b, overlap, rel_coordinates, direction):
    
    start = time.time()
    trim_a, trim_b = trimInput(img_a, img_b, overlap, direction)

    print(trim_a.shape)
    endTrim = time.time()
    correlation = phaseCorrelation(trim_a, trim_b)
    endCorrelation = time.time()
    print(correlation.shape)
    peaks = np.array(np.unravel_index(np.argmax(correlation.flatten()), correlation.shape))
    
    endDisplacement = time.time()
    ratio = correlation[rel_coordinates[0], rel_coordinates[1]]/correlation[peaks[0], peaks[1]]

    peaks[0] = peaks[0] if peaks[0] < int(correlation.shape[0]/2) else -(correlation.shape[0] - peaks[0])
    peaks[1] = peaks[1] if peaks[1] < int(correlation.shape[1]/2) else -(correlation.shape[1] - peaks[1])

    if np.all(peaks == rel_coordinates):
        # all coordinates are guessed correctly
        distance = [0, 0]
    else:
        distance = np.sqrt(np.abs(peaks**2 - rel_coordinates**2))

    # if np.any(distance > 20):
    #     print(distance)
    #     cv.imshow("a", trim_a)
    #     cv.imshow("b", trim_b)
    #     cv.imshow("1", img_a)
    #     cv.imshow("2", img_b)
    #     cv.waitKey(0)
    print(f"Guessed coords: {peaks}, real: {rel_coordinates}, dist: {distance}")

    full_time = endDisplacement - start
    corr_time = endCorrelation - endTrim
    trim_time = endTrim - start 
    print(ratio, len(np.where(correlation > correlation[rel_coordinates[0], rel_coordinates[1]])[0]))

    print()
    return np.array([full_time, corr_time, trim_time, distance[0], distance[1]])


def testImage(tiles, coordinates, relative_coordinates, overlap):

   

    results = np.zeros(((tiles.shape[0] - 1)*(tiles.shape[1]) + (tiles.shape[0])*(tiles.shape[1] - 1), 5), float)
    print(results.shape)
    k = 0
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            if i != tiles.shape[0] - 1:
                t = time_stitch(tiles[i, j], tiles[i+1, j], overlap[0], relative_coordinates[i+1, j][0], "UP")
                print(f"Stitch image [{i}, {j}] with [{i+1}, {j}] took {t[0]}s")
                results[k] = t
                k += 1
            if j != tiles.shape[1] - 1:
                t = time_stitch(tiles[i, j], tiles[i, j+1], overlap[1], relative_coordinates[i, j+1][1], "LEFT")
                print(f"Stitch image [{i}, {j}] with [{i}, {j+1}] took {t[0]}s")
                results[k] = t
                k += 1

    print(np.average(results, axis=0))
    print(f"max fulltime \t {np.amax(results[:, 0])}")
    print(f"max corrtime \t {np.amax(results[:, 1])}")
    print(f"max trimtime \t {np.amax(results[:, 2])}")
    print(f"max distance_y \t {np.amax(results[:, 3])}")
    print(f"max distance_x \t {np.amax(results[:, 4])}")
    

cv.imshow("im2", tiles[1, 1])
cv.imshow("im1", tiles[1, 2])
cv.waitKey(0)
cv.destroyAllWindows()

overlap = [50, 50]


testImage(tiles, c, c_r, overlap)








