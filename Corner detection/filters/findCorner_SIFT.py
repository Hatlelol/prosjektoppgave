import cv2 as cv
import numpy as np


from find_homography import *


debug = True

ransac_confidence = 0.85
ransac_inlier_threshold = 5.
save_image = False
use_matplotlib = False

lowerLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_000.ome.tif"
upperLeft = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos000_019.ome.tif"
upperRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_019.ome.tif"
lowerRight = R"C:\data\grid\pic_2\pic_2_MMStack_1-Pos014_000.ome.tif"
full = R"C:\data\grid\Fused_pic2_no_overlap_compute-jpg.jpg"

scene_img = cv.imread(lowerRight)[::2, ::2]
scene_img_gray = cv.cvtColor(scene_img, cv.COLOR_BGR2GRAY)
scene_img_gray = cv.bilateralFilter(scene_img_gray, d=9, sigmaColor=18, sigmaSpace=5)



object_img = cv.imread(R"C:\data\grid\corner.jpg")
object_img_gray = cv.cvtColor(object_img, cv.COLOR_BGR2GRAY)
object_img_gray = cv.rotate(object_img_gray, cv.ROTATE_90_COUNTERCLOCKWISE)
object_img_gray = cv.bilateralFilter(object_img_gray, d=9, sigmaColor=18, sigmaSpace=5)
object_img_gray[:530, :470] = np.mean(object_img_gray[:530, :470])

# object_img_gray = cv.GaussianBlur(object_img_gray, (7, 7), 0)
# object_img_gray = cv.rotate(object_img_gray, cv.ROTATE_90_COUNTERCLOCKWISE)

# Get SIFT keypoints_1 and descriptors
sift = cv.SIFT_create()
target_keypoints, target_descriptors = sift.detectAndCompute(scene_img_gray, None)
source_keypoints, source_descriptors = sift.detectAndCompute(object_img_gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

# For each keypoint in object_img get the 3 best matches
matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
matches = filter_matches(matches)

if debug:
    draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matches_img = cv2.drawMatches(object_img_gray,
                                    source_keypoints,
                                    scene_img_gray,
                                    target_keypoints,
                                    matches,
                                    None,
                                    **draw_params)
    show_image(matches_img, "Matches", save_image=False, use_matplotlib=False)

 # Convert keypoints arrays with shape (n, 2) in the OpenCV convention (x,y).
# source_points[i] is the matching keypoint to target_points[i]
source_points = np.array([source_keypoints[match.queryIdx].pt for match in matches])
target_points = np.array([target_keypoints[match.trainIdx].pt for match in matches])

# homography = find_homography_leastsquares(source_points, target_points)

homography, best_inliers, num_iterations = find_homography_ransac(source_points,
                                                                    target_points,
                                                                    confidence=ransac_confidence,
                                                                    inlier_threshold=ransac_inlier_threshold)

if debug:
    draw_params = dict(matchesMask=best_inliers.astype(int),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inlier_image = cv2.drawMatches(object_img_gray,
                                    source_keypoints,
                                    scene_img_gray,
                                    target_keypoints,
                                    matches,
                                    None,
                                    **draw_params)
    show_image(inlier_image, "Inliers", save_image=save_image, use_matplotlib=use_matplotlib)

# Plot results
plot_img = draw_rectangles(scene_img, object_img, homography)
show_image(plot_img, f"FinalResult", save_image=False, use_matplotlib=use_matplotlib)

transformed_object_img = cv2.warpPerspective(object_img, homography, dsize=scene_img.shape[1::-1])
scene_img_blend = scene_img.copy()
scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
show_image(scene_img_blend, "Overlay Object", save_image=save_image, use_matplotlib=use_matplotlib)


# print(np.amax(scene_img))
# cv.imshow("img", scene_img)
# cv.waitKey(0)
# cv.destroyAllWindows()