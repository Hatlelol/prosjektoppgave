import tifffile
import numpy as np
import os
import time
import cv2 as cv

def openFile(filename: str, key=None) -> np.ndarray:
    """
    Opens file with filename with the tifffile library
    """

    # return tifffile.imread(filename, key=key)
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)
    

def saveFile(filename: str, image: np.ndarray, photometric: str="rgb")-> None:
    """
    Saves the input image with filename with the tifffile library
    
    """
    if len(image.shape) == 2:
        photometric = "minisblack"
    
    tifffile.imwrite(filename, image, photometric=photometric)


def createFolder(source:str, folder_name:str) -> str:
    """
    Creates a new folder in the source directory. 

    source: source directory
    folder_name: folder name

    returns: the path to the folder.  
    
    """
    newpath = source + "\\" + folder_name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        return newpath
        
    print(f"Folder {newpath} already exist")
    return newpath
    


def deleteFolder(folder_name: str) -> None:
    """
    NOT IMPLEMENTED
    
    will delete the folder with folde_name
    """
    pass

#https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def applyNoise(rng: np.random.Generator, image: np.ndarray, vignett:bool = False, gaussian: bool = False, sAndp: bool = False, speckle: bool = False, poisson: bool = False):
    """
    Add noise by the modes listed. Adds them in a way that is natural for image aquisition.
    set each if the parameters to True to add the noise from them

    poisson noise is not fully operational
    
    """
    
    noise_types = {
        "specke": speckle,
        "vignett": vignett,
        "gaussian": gaussian,
        "poisson": poisson,
        "salt and peppet": sAndp
    }
    
    
    # rng = np.random.default_rng(42)
    if speckle:
    
        gauss = 0.01*np.random.randn(*image.shape)
        gauss = gauss.reshape(*image.shape)      
        image = np.where(image + (image * gauss).astype(np.uint8)>255, 255, image + (image * gauss).astype(np.uint8)) 

    if vignett:
        X_resultant_kernel = cv.getGaussianKernel(image.shape[1],image.shape[1])
        Y_resultant_kernel = cv.getGaussianKernel(image.shape[0],image.shape[0])
        
        #generating resultant_kernel matrix 
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

        #creating mask and normalising by using np.linalg
        # function
        mask = resultant_kernel / np.amax(resultant_kernel)

        image = (image*mask).astype(np.uint8)
        # image = (255/np.amax(image))*image
    if gaussian:
        gauss = rng.normal(0, np.sqrt(255*0.01), image.shape)
        over = image + gauss > 255
        under = image - gauss > 255
        image = np.where(np.logical_or(over, under), image, (image + gauss)).astype(np.uint8)

   
    if poisson:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = (rng.poisson(image * vals) / float(vals)).astype(np.uint8)
    if sAndp:
        s_vs_p = 0.5
        amount = 0.001
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        y_coords = np.random.randint(0, image.shape[0]-1, int(num_salt))
        x_coords = np.random.randint(0, image.shape[1]-1, int(num_salt))

        image[y_coords, x_coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        y_coords = np.random.randint(0, image.shape[0]-1, int(num_pepper))
        x_coords = np.random.randint(0, image.shape[1]-1, int(num_pepper))
    
        image[y_coords, x_coords] = 0
    return image, noise_types
    
    




def splitImage_old(image: np.ndarray, overlap_w: int, overlap_h: int, rows: int, cols: int, shift_w: int, shift_h: int, randomShift: int = 5, noise=False) -> (np.ndarray, np.ndarray):
    """
    Splits the image input into rows*cols images with the overlap and shift specified

    image: the input image to split
    overlap_w: overlap in pixels between the images in the vertical direction
    overlap_h: overlap in pixels between the images in the horizontal direction
    
    rows: Number of rows of images
    cols: Number of columns of images
    shift_w: Shift in the vertical direction between rows
    shift_h: Shift in the horizontal direction between columns
    randomShift: maximum pm random deviation each image has from the initial position
    noise: Bool to say if each resulting image will get noisy

    returns:
    tiles: np.ndarray of equal size images with shape (rows, cols, N, M)
    tiles_metadata: the metadata of the tiles, i.e the coordinates of each image
    
    """
    rng = np.random.default_rng(42)
    if randomShift < 1:
        x_random = np.zeros((rows, cols), dtype=np.uint8)
        y_random = np.zeros((rows, cols), dtype=np.uint8)
    else:
        x_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        y_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        x_random[0] = 0
        y_random[0] = 0

    w = image.shape[1]
    h = image.shape[0]
    


    # pad image with zeros according to shift

    
    output_w = int(w/cols + (overlap_w/cols)*(cols-1))
    output_h = int(h/rows + (overlap_h/rows)*(rows-1))
    j = cols -1 
    i = rows - 1
    print(w, h, abs(min(shift_h*cols, 0 )) + output_w*j + shift_h*i - overlap_w*j + x_random[i, j] + output_w, 
          abs(min(shift_w*rows, 0)) + output_h*i - overlap_h*i + shift_w*j + y_random[i, j] + output_h)



    #Pad images: 
    if (shift_w != 0 or shift_h != 0):
        # pad_s = min(0, shift_w*(rows))
        pad_s = abs(min(shift_h*cols, 0 )) + output_w*j + shift_h*i - overlap_w*j + x_random[i, j] + output_w - w
        # pad_w = max(0, shift_h*(cols)) 
        pad_w = 0
        # pad_n = max(0, shift_w*(rows)) 
        pad_n = 0
        # pad_e = min(0, shift_h*(cols)) 
        pad_e = abs(min(shift_w*rows, 0)) + output_h*i - overlap_h*i + shift_w*j + y_random[i, j] + output_h - h

        image  = np.pad(image, ((pad_n, pad_s), (pad_w, pad_e)), 'constant', constant_values=(0, 0))
    print(image.shape)
    print(image.shape[0] - h)
    print(image.shape[1] - w)

    if len(image.shape) == 3:
        tiles = np.zeros((rows, cols, output_h, output_w, 3), dtype=image.dtype)
    else:
        tiles = np.zeros((rows, cols, output_h, output_w), dtype=image.dtype)
    
    tiles_metadata = np.full((rows, cols, 2), 0, int)

    print(image.shape, output_h, output_w)
    print(x_random[:, -1])
 
    j = cols - 1
    i = rows - 1
    print(abs(min(shift_h*cols, 0 )), output_w*j,  shift_h*i, -overlap_w*j, abs(min(shift_h*cols, 0 )) + output_w*j + shift_h*i - overlap_w*j)

    print(abs(min(shift_w*rows, 0)), output_h*i, shift_w*j, - overlap_h*i, abs(min(shift_w*rows, 0)) + output_h*i  + shift_w*j - overlap_h*i)
    print(y_random[-1, :])


    for i in range(rows):
        y_start_row = abs(min(shift_w*rows, 0)) + output_h*i - overlap_h*i
        # y_end_row = y_start_row + output_h
        for j in range(cols):
            x_start = abs(min(shift_h*cols, 0 )) + output_w*j + shift_h*i - overlap_w*j + x_random[i, j]
            x_end = x_start + output_w

            y_start = y_start_row + shift_w*j + y_random[i, j]
            y_end = y_start + output_h

            print(f"{i}, {j}, {image[y_start + shift_w*j:y_end+shift_w*j, x_start:x_end].shape}, {output_h*i}, {output_w*j}, {y_start}, {x_start}")
            tile = image[y_start+shift_w*j:y_end+shift_w*j, x_start:x_end]
            
            if noise:
                tile = applyNoise(tile, vignett=True, gaussian=True, speckle=True)

            tiles_metadata[i, j] = (y_start, x_start)
            tiles[i, j] = tile

    return tiles, tiles_metadata


def splitImageSafe(image: np.ndarray, overlap_p: int, output_h: int, output_w: int, shift_w: int, shift_h: int, randomShift: int = 5, noise=False, debug=False) -> (np.ndarray, np.ndarray):
    
    w = image.shape[1]
    h = image.shape[0]
    
    cols = int(np.floor((w - randomShift - output_w*(overlap_p/100))/(output_w*(1-overlap_p/100)+ abs(shift_h))))
    rows = int(np.floor((h - randomShift - output_h*(overlap_p/100))/(output_h*(1-overlap_p/100)+ abs(shift_w))))
  

    rng = np.random.default_rng(42)
    if randomShift < 1:
        x_random = np.zeros((rows, cols))
        y_random = np.zeros((rows, cols))
    else:
        x_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        y_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        x_random[0] = 0
        y_random[0] = 0

    

    overlap_w = int(output_w*(overlap_p/100))
    overlap_h = int(output_h*(overlap_p/100))


    if len(image.shape) == 3:
        tiles = np.zeros((rows, cols, output_h, output_w, 3), dtype=image.dtype)
    else:
        tiles = np.zeros((rows, cols, output_h, output_w), dtype=image.dtype)
    
    tiles_metadata = np.full((rows, cols, 2), 0, int)

    noise_types=None
    

    for i in range(rows):
        y_start_row = abs(min(shift_w*(cols - 1), 0)) + output_h*i - overlap_h*i
        # y_end_row = y_start_row + output_h
        for j in range(cols):
            x_start = abs(min(shift_h*(rows - 1), 0 )) + output_w*j + shift_h*i - overlap_w*j #+ x_random[i, j]
            x_end = x_start + output_w

            y_start = y_start_row + shift_w*j #+ y_random[i, j]
            y_end = y_start + output_h
            if debug:
                print(f"{i}, {j}, {image[y_start + shift_w*j:y_end+shift_w*j, x_start:x_end].shape}, {output_h*i}, {output_w*j}, {y_start}, {x_start}")
            tile = image[y_start:y_end, x_start:x_end]
            
            if noise:
                tile, noise_types = applyNoise(rng, tile, vignett=True, gaussian=True, speckle=True)

            tiles_metadata[i, j][0] = y_start
            tiles_metadata[i, j][1] = x_start
            tiles[i, j] = tile

    return tiles, tiles_metadata, noise_types


def splitImage(image: np.ndarray, overlap_p: int, rows: int, cols: int, shift_w: int, shift_h: int, randomShift: int = 5, noise=False, debug=False) -> (np.ndarray, np.ndarray):
    """
    Splits the image input into rows*cols images with the overlap and shift specified

    image: the input image to split
    overlap_p: overlap in percent between images
    
    rows: Number of rows of images
    cols: Number of columns of images
    shift_w: Shift in the vertical direction between rows
    shift_h: Shift in the horizontal direction between columns
    randomShift: maximum pm random deviation each image has from the initial position
    noise: Bool to say if each resulting image will get noisy

    returns:
    tiles: np.ndarray of equal size images with shape (rows, cols, N, M)
    tiles_metadata: the metadata of the tiles, i.e the coordinates of each image
    
    """
    rng = np.random.default_rng(42)
    if randomShift < 1:
        x_random = np.zeros((rows, cols))
        y_random = np.zeros((rows, cols))
    else:
        x_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        y_random = rng.integers(low=-randomShift, high=randomShift, size=(rows, cols))
        x_random[0] = 0
        y_random[0] = 0

    w = image.shape[1]
    h = image.shape[0]
    
    full_shift_w = abs(shift_w*(cols))
    full_shift_h = abs(shift_h*(rows))
    
    w_r = w - full_shift_w - randomShift
    h_c = h - full_shift_h - randomShift


    output_w = int(np.floor(w_r/(cols - (overlap_p/100)*(cols - 1))))
    output_h = int(np.floor(h_c/(rows - (overlap_p/100)*(rows - 1))))


    overlap_w = int(output_w*(overlap_p/100))
    overlap_h = int(output_h*(overlap_p/100))


    if len(image.shape) == 3:
        tiles = np.zeros((rows, cols, output_h, output_w, 3), dtype=image.dtype)
    else:
        tiles = np.zeros((rows, cols, output_h, output_w), dtype=image.dtype)
    
    tiles_metadata = np.full((rows, cols, 2), 0, int)

    noise_types=None
    if noise:
        rng = np.random.default_rng(42)



    for i in range(rows):
        y_start_row = abs(min(shift_w*(cols - 1), 0)) + output_h*i - overlap_h*i
        # y_end_row = y_start_row + output_h
        for j in range(cols):
            x_start = abs(min(shift_h*(rows - 1), 0 )) + output_w*j + shift_h*i - overlap_w*j #+ x_random[i, j]
            x_end = x_start + output_w

            y_start = y_start_row + shift_w*j #+ y_random[i, j]
            y_end = y_start + output_h
            if debug:
                print(f"{i}, {j}, {image[y_start + shift_w*j:y_end+shift_w*j, x_start:x_end].shape}, {output_h*i}, {output_w*j}, {y_start}, {x_start}")
            tile = image[y_start:y_end, x_start:x_end]
            
            if noise:
                tile, noise_types = applyNoise(rng, tile, vignett=True, gaussian=True, speckle=True)

            tiles_metadata[i, j][0] = y_start
            tiles_metadata[i, j][1] = x_start
            tiles[i, j] = tile

    return tiles, tiles_metadata, noise_types, overlap_w, overlap_h

def drawSplits(input_image, rows, cols, overlap_p, shift_w, shift_h, random_shift, arrow="None"):
    """
    Function for visualizing the splits in the image. 

    input_image: the image to draw the input on
    rows: number of rows the image will be split into
    cols: number of columns the image will be split into
    overlap_p: the percentage of overlap between images
    shift_w: the shift from row to row
    shift_h: the sift from column to column
    random_shift: the maximum amount of pm random shift to be added to each image.     
    arrow: if we want to draw an arrow in the direction or not
    """
    rng = np.random.default_rng(42)
    if random_shift < 1:
        x_random = np.zeros((rows, cols), dtype=np.uint8)
        y_random = np.zeros((rows, cols), dtype=np.uint8)
    else:
        x_random = rng.integers(low=-random_shift, high=random_shift, size=(rows, cols))
        y_random = rng.integers(low=-random_shift, high=random_shift, size=(rows, cols))
        x_random[:, 0] = 0
        y_random[0, :] = 0

    w = input_image.shape[1]
    h = input_image.shape[0]
    
    full_shift_w = shift_w*(cols)
    full_shift_h = shift_h*(rows)
    
    w_r = w - full_shift_w - random_shift
    h_c = h - full_shift_h - random_shift



    output_w = int(np.floor(w_r/(cols - (overlap_p/100)*(cols - 1))))
    output_h = int(np.floor(h_c/(rows - (overlap_p/100)*(rows - 1))))



    overlap_w = int(output_w*(overlap_p/100))
    overlap_h = int(output_h*(overlap_p/100))
    colarr = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    k = cols*(rows - 1)
    output_image = cv.cvtColor(input_image, cv.COLOR_GRAY2BGR)
    c_i = 0
    arrow_arr = []
    for i in range(rows):
        y_start_row = abs(min(shift_w*cols, 0)) + output_h*i - overlap_h*i
        # y_end_row = y_start_row + output_h
        c = i
        for j in range(cols):
            
            c = colarr[c_i]

            x_start = abs(min(shift_h*rows, 0 )) + output_w*j + shift_h*i - overlap_w*j + x_random[i, j]
            x_end = x_start + output_w

            y_start = y_start_row + shift_w*j + y_random[i, j]
            y_end = y_start + output_h

            output_image[y_start:y_start + 3, x_start:x_end] = c
            output_image[y_end - 3:y_end, x_start:x_end] = c
            output_image[y_start:y_end, x_start:x_start + 3] = c
            output_image[y_start:y_end, x_end - 3:x_end] = c

            cv.putText(
            output_image, #numpy array on which text is written
            str(k), #text
            (x_start + int(output_w/2), y_start + int(output_h/2)), #position at which writing has to start
            cv.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            c, #font color
            3) #font stroke
            k+= 1
            c_i+= 1
            if c_i >= len(colarr):
                c_i = 0

            if arrow != "None":
                if arrow == "full-reverse":

                    if i == rows-1 and j == 0:
                        pass
                    elif (i + 1) % 2 == 0:
                        # mot høyre

                        if j == 0:
                            start_point = (y_end, x_start + int(output_w/2))
                            end_point = (y_end - overlap_h, x_start + int(output_w/2))
                        else:
                            start_point = (y_start + int(output_h/2), x_start)  
                            end_point = (y_start + int(output_h/2), x_start + overlap_w)  
                    else:
                        #mot venstre
                        if j == cols - 1:
                            start_point = (y_end, x_start + int(output_w/2))  
                            end_point = (y_end - overlap_h, x_start + int(output_w/2))  
                        else:
                            start_point = (y_start + int(output_h/2), x_end)  
                            end_point = (y_start + int(output_h/2), x_end - overlap_w)  
                    
                    # Red color in BGR  
                    color = c  
                    
                    # Line thickness of 9 px  
                    thickness = 9
                    arrow_arr.append([start_point, end_point, color])
                    # Using cv2.arrowedLine() method  
                    # Draw a red arrow line 
                    # with thickness of 9 px and tipLength = 0.5 
                    # cv.imshow(f"{i}, {j}", output_image)
                    # cv.waitKey(0)
                if arrow == "x":
                    
                    if j == 0 and i == rows -1:
                        pass
                    elif j == 0:
                        start_point = (y_end, x_start + int(output_w/2))
                        end_point = (y_end - overlap_h, x_start + int(output_w/2))
                    else:
                        start_point = (y_start + int(output_h/2), x_start)  
                        end_point = (y_start + int(output_h/2), x_start + overlap_w)  
                    
                    
                    # Red color in BGR  
                    color = c  
                    
                    # Line thickness of 9 px  
                    thickness = 9
                    arrow_arr.append([start_point, end_point, color])
                if arrow == "y":
                    
                    if j == 0 and i == rows - 1:
                        pass
                    elif i == rows - 1:
                        start_point = (y_start + int(output_h/2), x_start)  
                        end_point = (y_start + int(output_h/2), x_start + overlap_w)  
                    else:
                        start_point = (y_end, x_start + int(output_w/2))
                        end_point = (y_end - overlap_h, x_start + int(output_w/2))
                    
                    
                    # Red color in BGR  
                    color = c  
                    
                    # Line thickness of 9 px  
                    thickness = 9
                    arrow_arr.append([start_point, end_point, color])

        k -= 2*cols
    if arrow != "None":
        for a in arrow_arr:
            output_image = cv.arrowedLine(output_image, a[0][::-1], a[1][::-1],  
                            a[2], thickness, tipLength = 0.5)  

    cv.imshow("split in images", output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
image = np.full((1000, 1000), (126), dtype=np.uint8)

drawSplits(image, 4, 4, 20, 0, 0, 0, "x")
drawSplits(image, 4, 4, 20, 0, 0, 0, "y")

def combineImages(image_1, image_2, overlap, direction):
    img1 = openFile(image_1)
    img2 = openFile(image_2)
    if img1.shape != img2.shape:
        print("Image shapes are not equal")
        return
    shape = img1.shape
    # Normalize images
    img1_norm = ((img1/np.amax(img1))*255).astype(np.uint8)
    img2_norm = ((img2/np.amax(img2))*255).astype(np.uint8)


    start_divide = time.time()
    if direction == "up":
        image_full = np.zeros((shape[0]*2 - overlap, shape[1]), dtype=np.uint8)
        image_full[:shape[0], :] = img1_norm
        image_full[shape[0] - overlap:, :] = img2_norm
    elif direction == "right":
        image_full = np.zeros((shape[0], shape[1]*2 - overlap), dtype=np.uint8)
        image_full[:, :shape[1]] = img1_norm
        image_full[:, shape[1] - overlap:] = img2_norm
    time_divide = time.time() - start_divide

    stitch = stitchImages(image_1, image_2)
    result = evaluateStitch(image_full, stitch)


    







