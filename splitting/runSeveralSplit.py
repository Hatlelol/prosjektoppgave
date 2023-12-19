import subprocess
import os
from helpFunctions import *

def findShape(folder_name):
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
    img = openFile(filename)

    return img.shape

SHIFT_SCALE = 0.05 # 1/20
RANDOM_SHIFT_SCALE = 0.5*SHIFT_SCALE
OVERLAP = 15
ORIGIN = "LL"
NOISE = True

def runSplit(folder, num_y, num_x, inp_shape):
    script_path = "C:\data\space\splitsafe.py"

    script_arguments = [
        folder,
        str(num_y), 
        str(num_x), 
        str(OVERLAP), 
        str(int(inp_shape[0]*SHIFT_SCALE)),
        str(int(inp_shape[1]*SHIFT_SCALE)),
        ORIGIN,
        str(int(RANDOM_SHIFT_SCALE*max(inp_shape[0], inp_shape[1]))),
        str(NOISE)
        ]
    print(f"Split {folder}, ({num_y}, {num_x}), {inp_shape}")
    # Use subprocess to run the script with arguments
    subprocess.run(["python", script_path] + script_arguments)

if __name__ == "__main__":
    # Specify the path to the script and provide arguments
    

    folders = [
        "CosmicCliffs",
        "DeepField",
        "SouthernRing",
        "Stephans_Quintet"
    ]

    shapes = [findShape(folder) for folder in folders]

    min_x = 400
    min_y = 400

    ended = np.full(len(folders), False, bool)

    dims = 500

    while np.any(ended == False):
        print(ended)
        for i, f in enumerate(folders):
            if ended[i]:
                continue
            print(dims)
            print(shapes[i])
            print(shapes[i][0]/dims, shapes[i][1]/dims)
            print(shapes[i][0]/dims < min_y, shapes[i][1]/dims < min_x)
            print()

            x_dim, y_dim = dims, dims
            if shapes[i][0]/dims < 2:
                y_dim = int((shapes[i][0] // 1000)*1000/2)
            if shapes[i][1]/dims < 2:
                x_dim = int((shapes[i][1] // 1000)*1000/2)
            if x_dim != dims and y_dim != dims:
                ended[i] = True
                continue
 
            runSplit(f, y_dim, x_dim, shapes[i])

        if dims >= 3000:
            dims += 2000
        else:
            dims *= 2 
        
        
        

            
            
