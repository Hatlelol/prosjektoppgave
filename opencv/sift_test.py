import numpy as np
import cv2 as cv



gray_1 = cv.imread(R'C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos000_000.ome.jpg', cv.IMREAD_GRAYSCALE)
scale = (int(gray_1.shape[0]/2), int(gray_1.shape[1]/2))
gray_1 = cv.resize(gray_1, scale, interpolation= cv.INTER_LINEAR)
sift = cv.SIFT_create()
overlap = int(len(gray_1[0])*(1-0.1))
kp1, des1 = sift.detectAndCompute(gray_1[:, overlap:],None)

for i in kp1:
    i.pt = (i.pt[0] + overlap, i.pt[1])
img=cv.drawKeypoints(gray_1,kp1,gray_1)
# cv.imwrite(R'C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\out.jpg',img)
gray_2 = cv.imread(R'C:\Users\sondr\OneDrive - NTNU\KodeTesting\python\image_stitching\out\dsd_10_MMStack_1-Pos001_000.ome.jpg', cv.IMREAD_GRAYSCALE)
scale = (int(gray_2.shape[0]/2), int(gray_2.shape[1]/2))
gray_2 = cv.resize(gray_2, scale, interpolation= cv.INTER_LINEAR)

sift = cv.SIFT_create()
overlap = int(len(gray_2[0])*(0.1))
kp2, des2 = sift.detectAndCompute(gray_2[:, :overlap],None)

# for i in kp:
#     i.pt = (i.pt[0] + overlap, i.pt[1])
img_2=cv.drawKeypoints(gray_2,kp2,gray_2)







cv.imshow("img1",img)
cv.imshow("img2",img_2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)  

MIN_MATCH_COUNT = 1

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    print(src_pts)
    print(dst_pts)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    # print(img.shape)
    # h,w = gray_1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)
    # gray_2 = cv.polylines(gray_2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

x_diff = []
y_diff = []
for a, b in zip(src_pts, dst_pts):
    x_diff.append(a[0][1] - b[0][1])
    y_diff.append(a[0][0] - b[0][0])
    print(f"Point a to b is: {a} {b}")

x_diff = np.array(x_diff)
y_diff = np.array(y_diff)
print(f"Average x_diff = {np.average(x_diff)}, var: {np.var(x_diff)}")
print(f"Average y_diff = {np.average(y_diff)}, var: {np.var(y_diff)}")

y_move = int(round(np.mean(x_diff)))
x_move = int(round(np.mean(y_diff)))

out = np.zeros((gray_1.shape[0] + abs(y_move), x_move + gray_1.shape[1]), dtype=np.uint8)
print(out.shape)
print(gray_1.shape, gray_2.shape)
print(y_move, x_move)
out[abs(y_move):abs(y_move) + gray_1.shape[0], :gray_1.shape[1]] = gray_1
out[:gray_1.shape[0], x_move:x_move + gray_1.shape[1]] = gray_2




draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)
img3 = cv.drawMatches(gray_1,kp1,gray_2,kp2,good,None,**draw_params)
cv.imshow("img3", img3)

cv.imshow("OUt", out)
cv.waitKey(0)
cv.destroyAllWindows()