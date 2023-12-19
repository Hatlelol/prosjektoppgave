import sys
import os
from helpFunctions import *

def inpToBool(inp):
    if str(inp) == "0":
        return False
    if str(inp) == "1":
        return True
    if str(inp) == "True":
        return True
    return False

def writeTiles(tiles, metadata, overlap, inp_shape, folder, origin="UL", noise_types=None):
    """
    writes the tiles to images, and writes the metadata file.

    tiles: the images in a ROWSxCOLS matrix
    metadata: metadata of the images, i.e the coords
    overlap: overlap in percent
    folder: folder to write the files to
    origin: what corner will have the (0, 0) tile
    """

    rows = tiles.shape[0]
    cols = tiles.shape[1]
    noise_str = ""
    if noise_types is not None:
        for key, value in noise_types.items():
           if value:
               noise_str += f"{key}, " 

        noise_str = noise_str[:-2]


    randomShift = 5

    if origin is None or not (origin == "LL" or origin == "UR" or origin == "LR" or origin == "UL"):
        origin = "UL"

    x = lambda x:x
    y = lambda y:y
    c_x = lambda x, c:x
    c_y = lambda y, c:y
    if origin == "LL":
        x = lambda x:x
        y = lambda y:rows - y - 1

        c_x = lambda x, c:x
        if rows > 1:
            c_y = lambda y, c:metadata[rows-1, c][0] - y
        else:
            c_y = lambda y, c: -y
    elif origin == "UR":
        x = lambda x:cols - x - 1
        y = lambda y:y
        if cols > 1:
            c_x = lambda x, c: metadata[c, cols - 1][1] - x
        else:
            c_x = lambda x, c: -x
        c_y = lambda y, c: y
    elif origin == "LR":
        x = lambda x:cols - x - 1
        y = lambda y:rows - y - 1

        if cols > 1:
            c_x = lambda x, c: metadata[c, cols - 1][1] - x
        else:
            c_x = lambda x, c: -x
        if rows > 1:
            c_y = lambda y, c:metadata[rows-1, c][0] - y
        else:
            c_y = lambda y, c: -y
    
    out_string = f"origin: {origin}\noverlap_h: {tiles[0, 0].shape[0]*overlap/100} overlap_v: {tiles[0, 0].shape[1]*overlap/100}\n\n"
    out_string_altered = f"origin: {origin}\noverlap_h: {tiles[0, 0].shape[0]*overlap/100} overlap_v: {tiles[0, 0].shape[1]*overlap/100}\n\n"

    rng = np.random.default_rng(12)
    
    x_random = randomShift*(2*rng.random((rows, cols)) - 1)
    y_random = randomShift*(2*rng.random((rows, cols)) - 1)
    
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            out_filename = f"{folder}\img_{y(i):03}_{x(j):03}.tif"
            saveFile(out_filename, tiles[i, j])

            coords = metadata[y(i), x(j)]
            out_string          += f"{folder}\img_{i:03}_{j:03}.tif ;; ({c_y(coords[0], j)}, {c_x(coords[1], i)})\n"
            out_string_altered  += f"{folder}\img_{i:03}_{j:03}.tif ;; ({c_y(coords[0] + y_random[i, j], j)}, {c_x(coords[1] + x_random[i, j], i)})\n"

    with open(f'{folder}\parameters.txt', 'w') as output:
        output.write(f"""
input image size: \t{inp_shape},\n
tile size: \t{tiles[0, 0].shape},\n
Overlap percent: \t{overlap},\n
Overlap in pixels(y, x): \t({tiles[0, 0].shape[0]*overlap/100}, {tiles[0, 0].shape[1]*overlap/100}),\n
origin: \t {origin},\n
noise types: \t {noise_str}
""")

    with open(f'{folder}\metadata.txt', 'w') as output:   
        output.write(out_string)
    with open(f'{folder}\metadata_altered.txt', 'w') as output:   
        output.write(out_string_altered)





def main(folder_name, rows, cols, overlap_p, shift_w, shift_h, origin, random_shift, noise=False):
    """
    folder_name: name of the folder that has the images
    rows: number of rows to split the image into
    cols: number of collumns to spit the image into
    overlap_p: the overlap percentage between the images (0-100)
    shift_w: the vertical shift per image row
    shift_h: the horizontal shift per image column
    origin: which corner the the image 0, 0 will have. default = "UL"    
    random_shift: if there will be random shift between images
    noise: if there will be noise added on the images
    """
    start_time = time.time()
    folder = os.path.join(os.getcwd(), folder_name)
    filename = ""
    for file in os.listdir(folder):
        # filename = os.fsdecode(file)
        if file.endswith(".tif") or file.endswith(".tiff") or file.endswith(".png") or file.endswith(".jpg"): 
            #take first image file in folder and use that
            filename = os.path.join(folder, file)
            break

    if filename == "":
        print(f"Did not find a tif, jpg or png file in folder {folder}")
        return 
    print(f"read file {filename} in folder {folder}")

    print(f"Time used for finding file {time.time() - start_time} ")
    start_time = time.time()
    
    source_image = openFile(filename)
    shape = source_image.shape
    print(f"Time used for opening file {time.time() - start_time} ")
    start_time = time.time()

    if folder_name == "Trondheim":
        max_img = np.amax(source_image)
        min_img = np.amin(source_image)

        source_image_temp = ((source_image.astype(np.float32) - min_img)/(max_img - min_img))
        source_image = (255*source_image_temp).astype(np.uint8)
        


    # ovrelap_w = int((shape[1]/cols)*(overlap_p/100)/(1 - overlap_p/100))
    # overlap_h = int((shape[0]/rows)*(overlap_p/100)/(1 - overlap_p/100))
    
    # if random_shift:
    #     random_shift = int(min((shape[1]/cols)*0.005, (shape[0]/rows)*0.005))
    # else:
    #     random_shift = 0

    tiles, metadata, noise_types = splitImageSafe(source_image, overlap_p, rows, cols, shift_w, shift_h, randomShift=random_shift, noise=noise)
    print(f"Time used for splitting {time.time() - start_time} ")
    start_time = time.time()

    
    save_folder = createFolder(folder, f"split_{rows}_{cols}")
    writeTiles(tiles, metadata, overlap_p, source_image.shape, save_folder, origin, noise_types)
    print(f"Time used for writing {time.time() - start_time} ")
    start_time = time.time()
    print(f"Write files to the folder {save_folder}\n")






if __name__ == "__main__":
    
    print(len(sys.argv))
    folder_name = "CosmicCliffs" 
    rows_w = 1000 
    col_h = 1000 
    overlap_p = 10 
    shift_h = 30 
    shift_w = 30 
    origin = "LL" 
    random_shift = True 
    noise = True

    # folder_name  = None
    # rows = None
    # cols = None
    # overlap_p = None
    # shift_w = None
    # shift_h = None
    # origin = None
    # random_shift = None
    # noise = None
    if len(sys.argv) < 4:
        print("Not enough parameters are defined")
        print("commands is folder_name, rows, cols, overlap_p, shift_w, shift_h, origin, random_shift, noise")
    else:
        folder_name = str(sys.argv[1])
        rows_w = int(sys.argv[2])
        col_h = int(sys.argv[3])
    
    
    if len(sys.argv) >= 10:
        noise = inpToBool(sys.argv[9])
    if len(sys.argv) >= 9:
        random_shift = int(sys.argv[8])
    if len(sys.argv) >= 8:
        origin = str(sys.argv[7])
    if len(sys.argv) >= 7:
        shift_h = int(sys.argv[6])
    if len(sys.argv) >= 6:
        shift_w = int(sys.argv[5])
    if len(sys.argv) >= 5:
        overlap_p = int(sys.argv[4])
   
    


    if folder_name != None:
        if rows_w != None:
            if col_h != None:
                if overlap_p == None:
                    print("Overlap percentage is none, setting to 0")
                    overlap_p = 0
                    shift_w = 0
                    shift_h = 0
                    origin = "UL"
                    random_shift = 0
                    noise = False
                elif shift_w == None:
                    print("Shift_w is not defined, setting zero")
                    shift_w = 0
                    shift_h = 0
                    origin = "UL"
                    random_shift = 0
                    noise = False
                elif shift_h == None:
                    print("shift_h is not defined, setting zero")
                    shift_h = 0
                    origin = "UL"
                    random_shift = 0
                    noise = False
                elif origin == None:
                    print("Origin is not defined, setting to default (UL)")
                    origin = "UL"
                    random_shift = 0
                    noise = False
                elif random_shift is None:
                    random_shift = 0
                    noise = False
                elif noise is None:
                    noise = False

                print(f"Running split progran with parameters {folder_name} {rows_w} {col_h} {overlap_p} {shift_w} {shift_h} {origin} {random_shift} {noise}")

                main(folder_name=folder_name, rows=rows_w, cols=col_h, overlap_p=overlap_p, shift_w=shift_w, shift_h=shift_h, origin=origin, random_shift=random_shift, noise=noise)
                
            else:
                print("No col_h are defined, commands is folder_name, rows_w, col_h, overlap_p, shift_w, shift_h, origin")
        else:
            print("No rows_w are defined, commands is folder_name, rows_w, col_h, overlap_p, shift_w, shift_h, origin")
    else:
        print("No folder name is defined, commands is folder_name, rows_w, col_h, overlap_p, shift_w, shift_h, origin")


