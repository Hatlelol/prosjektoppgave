import numpy as np
from tifffile import astype



def relativeToAbsolute_scrap(coords, direction, img_shape):
    """
    Converts the coordinates from relative (every single coordinat erelative to the last)
    to absolute (absolute in large grid)

    coords
    direction = "x_normal" "x_alternate" "y_normal" "y_alternate"
    
    """

    prev_row = None
    prev_col = None

    row = 0
    col = 0
    
    if direction == "x_normal":
        end = coords.shape[:2]
    elif direction == "x_alternate":
        end = coords.shape[:2] if coords.shape[0] %2 != 0 else (coords.shape[0], 0)
    elif direction == "y_normal":
        end = coords.shape[:2]
    elif direction == "y_alternate":
        end = coords.shape[:2] if coords.shape[1] % 2 != 0 else (0, coords.shape[1])

    else:
        end = coords.shape[:2]

    while row != end[0] - 1 or col != end[1] - 1:
        if prev_row is not None or prev_col is not None:
            if direction == "x_normal":
                print( coords[prev_row, prev_col] + coords[row, col]  )
                coords[row, col] = coords[prev_row, prev_col] + coords[row, col]   
            elif direction == "x_alternate":
                if row % 2 == 0 or col == prev_col:
                    coords[row, col] = coords[prev_row, prev_col] + coords[row, col]   
                else:
                    coords[row, col][0] = coords[prev_row, prev_col][0] + coords[row, col][0]
                    coords[row, col][1] = coords[prev_row, prev_col][1] - coords[row, col][1]
                    
            elif direction == "y_normal":
                print( coords[prev_row, prev_col] + coords[row, col]  )
                coords[row, col] = coords[prev_row, prev_col] + coords[row, col]   
            elif direction == "y_alternate":
                if col % 2 == 0 or row == prev_row:
                    coords[col, row] = coords[prev_col, prev_row] + coords[col, row]   
                else:
                    coords[col, row][0] = coords[prev_col, prev_row][0] + coords[col, row][0]
                    coords[col, row][1] = coords[prev_col, prev_row][1] - coords[col, row][1]
        

        prev_col = col
        prev_row = row

        if direction == "x_normal":
            if col == coords.shape[1] - 1:
                prev_col = 0
                prev_row = row
                col = 0
                row += 1
            else:
                col += 1            
        elif direction == "x_alternate":
            if row % 2 == 0:
                if col == coords.shape[1] - 1:
                    row += 1
                else:
                    col += 1
            else:
                if col == 0:
                    row += 1
                else:
                    col -= 1
        elif direction == "y_normal":
            if row == coords.shape[0] - 1:
                prev_row = 0
                prev_col = col
                row = 0
                col += 1
            else:
                row += 1
        elif direction == "y_alternate":
            if col % 2 == 0:
                if row == coords.shape[0] - 1:
                    col += 1
                else:
                    row += 1
            else:
                if row == 0:
                    col += 1
                else:
                    row -= 1

    return coords   

def relativeToAbsolute(coords, direction):


    if direction == "x_normal":
        for i in range(coords.shape[0]):
            if i != 0:
                coords[i, 0] = coords[i - 1, 0] + coords[i, 0]
            for j in range(1, coords.shape[1]):
                coords[i, j] = coords[i, j - 1] + coords[i, j]

    if direction == "y_normal":
        for j in range(coords.shape[1]):
            if j != 0:
                coords[0, j] = coords[0, j-1] + coords[0, j]
            for i in range(1, coords.shape[0]):
                print(coords[i - 1, j],  coords[i, j])
                coords[i, j] = coords[i - 1, j] + coords[i, j]

    return coords




# test_arr = np.array([[[0, 0], [4, 4], [-4, 4]], [[9, 2], [3, 3], [5, 3]]])

# print(relativeToAbsolute(test_arr, "x_normal", 0))



def getOutputDimensions(coords, image_shape):
        x_maks = x_min = y_maks = y_min = 0
        rows = coords.shape[0]
        cols = coords.shape[1]
        for i in range(rows):
            for j in range(cols):

                if j != 0 and j != cols - 1 and i != 0 and i != rows-1:
                    continue
                p = coords[i, j]
                print(p)
                y = p[0] if p[0] != np.inf else 0
                x = p[1] if p[1] != np.inf else 0
                if x + image_shape[1] > x_maks:
                    x_maks = x + image_shape[1]
                if x < x_min:
                    x_min = x
                if y + image_shape[0] > y_maks:
                    y_maks = y + image_shape[0]
                if y < y_min:
                    y_min = y
        return x_maks, x_min, y_maks, y_min

def generateOutput(images, coords, distance=None, vignett_inv=None, border = False):

    if distance is None and vignett_inv is None:
        transform = lambda image: image.astype(np.uint32)
    elif distance is None:
        transform = lambda image: (image*vignett_inv).astype(np.uint32)
    elif vignett_inv is None:
        transform = lambda image: (image*distance).astype(np.uint32)
    else:
        transform = lambda image: (image*vignett_inv*distance).astype(np.uint32)


    shape = images[0, 0].shape
    x_maks, x_min, y_maks, y_min = getOutputDimensions(coords, shape)

    out = np.zeros((abs(y_min) + y_maks, abs(x_min) + x_maks), dtype=np.uint32)
    if distance is not None:
        out_d = out.copy()

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            dy = coords[i,j][0] if coords[i,j][1] != np.inf else 0
            dx = coords[i,j][1] if coords[i,j][0] != np.inf else 0
            f_y = y_maks - dy - shape[0]
            f_x = abs(x_min) + dx
            # print(dx, dy)
            out[f_y:f_y + shape[0], f_x:f_x + shape[1]] += transform(images[i, j])
            if distance is not None:
                out_d[f_y:f_y + shape[0], f_x:f_x + shape[1]] += distance
            if border:
                out[f_y-10, f_x - 100: f_x + shape[1] + 100] = 255
                out[f_y+10, f_x - 100: f_x + shape[1] + 100] = 255
                out[f_y - 100:f_y + shape[0] + 100, f_x-10] = 255
                out[f_y - 100:f_y + shape[0] + 100:, f_x+10] = 255
                if distance is not None:
                    out_d[f_y-10, f_x - 100: f_x + shape[1] + 100] = 1
                    out_d[f_y+10, f_x - 100: f_x + shape[1] + 100] = 1
                    out_d[f_y - 100:f_y + shape[0] + 100, f_x-10] = 1
                    out_d[f_y - 100:f_y + shape[0] + 100:, f_x+10] = 1
    if distance is not None:
        out = out / out_d
    print(f"out_dimensions is max: {np.amax(out)}, min: {np.amin(out)}")
    return out.astype(np.uint8)

import cv2 as cv
def showResults(images_results, method):
    """
    Shows the results from the images in results, both x and y with the desired method.
    
    """
    
    res_x = images_results["result_x"]
    res_y = images_results["result_y"]
    image = np.zeros((res_x.shape[0], res_x.shape[1]+1, res_x[0, 0]["image1"].shape[0], res_x[0, 0]["image1"].shape[1]), np.uint8)
    
    print(res_x.shape)
    print(res_y.shape)
    x_coords = np.zeros((res_x.shape[0], res_x.shape[1] + 1, 2), np.int16)
    for i in range(res_x.shape[0]):
        for j in range(res_x.shape[1]):
            image[i, j] = res_x[i, j]["image1"]
            if j == res_x.shape[1] - 1:
                image[i, j+1] = res_x[i, j]["image2"]

            x_coords[i, j+1] = res_x[i, j]["relative position"] - res_x[i, j][method].getResult()

   
    y_coords = np.zeros((res_y.shape[0] + 1, res_y.shape[1], 2), np.int16)
    for i in range(res_y.shape[0]):
        for j in range(res_y.shape[1]):
            y_coords[i+1, j] = res_y[i, j]["relative position"] - res_y[i, j][method].getResult()



    x_coords[1:, 0] = y_coords[1:, 0]

    y_coords[0, 1:] = x_coords[0, 1:]

    x_coords = relativeToAbsolute(x_coords, "x_normal")
    y_coords = relativeToAbsolute(y_coords, "y_normal")
    img_x = generateOutput(image, x_coords)
    cv.imshow(f"x_dir_{method}", img_x)
    cv.waitKey(0)

    print(y_coords)
    img_y = generateOutput(image, y_coords)
    cv.imshow(f"y_dir_{method}", img_y)
    cv.waitKey(0)

    cv.destroyAllWindows()


def findImageResult(results, folder_name, split_x, split_y = None, method="normal"):
    """
    From a result list with dicionaries for the folder tests print the resulting image

    results: list of dictionaries with the result of the split of that dict
    folder_name: name of the folder that was stitched, will look for in results["name"] == folder_name
    split_x: the split in the x-direction of the images,
    split_y: split in the y-direction of the images, if None split_y = split_x
    method: the method to plot for "normal", "template" "SIFT" etc. 
    
    """

    split_y = split_x if split_y is None else split_y
    goal_dict = None
    for d in results:
        if d["name"] == folder_name and d["split"] == split_x:
            goal_dict = d        
            break
    if goal_dict is None:
        print(f"Did not find {folder_name} with {split_x}, {split_y} in result")
        return None

    showResults(goal_dict, method)

