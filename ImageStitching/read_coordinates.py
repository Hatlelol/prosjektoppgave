
import re
import numpy as np


def openFileAndFindMatches(filename, regex, flags=0):
    with open(filename) as f:
        data = f.read()

    if data is None:
        print(f"Could not find file {filename}")
        return []
    
    s = re.findall(regex, data, flags=flags)

    if s is None:
        print(f"Found no matches in file {filename} with regex {regex}")
        return []
    
    return s

def readMetadataFile(filename, image_regex=None):
#for reading metadata files
# filename = ""
    if image_regex is None:
        image_regex = r'"Label": "([\w-]+)",.*?"GridRow": (\d+).+?"GridCol": (\d+).+?"Position_um": \[\n +([\d\-.]+),\n +([\d\-.]+)'
    s = openFileAndFindMatches(filename, regex=image_regex, flags=re.DOTALL)
    if len(s) == 0:
        return s

    files = []

    for match in s:
        
        Label = match[0]
        Row = int(match[1])
        Col = int(match[2])
        x_pos = float(match[3])
        y_pos = float(match[4])
        
        pos =  re.search(r'1-Pos(\d{3})_(\d{3})', Label)
        if pos is None:
            print("Label template not correct.")
        else:
            if Row != int(pos.group(2)):
                print(f"Wrong row for label {Label}, {Row} {pos.group(2)}")
            if Col != int(pos.group(1)):
                print(f"Wrong col for label {Label}, {Col} {pos.group(1)}")

        obj = {
            "label": Label,
            "index": (Row, Col),
            "pos": np.array([x_pos, y_pos])
        }
        files.append(obj)

    return files

def readTileConfigurationCoordinates(filename, coordinate_regex=None, indecies=True):
    if coordinate_regex is None:
        # coordinate_regex = r'\(([-\d\.]+), ([-\d\.]+)\)
        if indecies:
           coordinate_regex = r"(dsd_10_MMStack_1-Pos(?P<COL>\d{3})_(?P<ROW>\d{3}).ome.tif *; *; *\((?P<YPOS>[-\d\.]+), (?P<XPOS>[-\d\.]+)\))" 
        else:
            coordinate_regex = r"(\((?P<YPOS>[-\d\.]+), (?P<XPOS>[-\d\.]+)\))"

    s = openFileAndFindMatches(filename, coordinate_regex)
    if len(s) == 0:
        return s
    if indecies:
        output= np.zeros((len(s), 4), dtype=int)
    else:
        output = np.zeros((len(s), 2), dtype=float)
    
    for i, match_init in enumerate(s):
        # print(match_init)
        match = re.search(coordinate_regex, match_init[0])
        # print(match)
        if indecies:
            output[i] = np.array([int(match.group("ROW")), int(match.group("COL")), float(match.group("YPOS")), float(match.group("XPOS"))])
        else:
            output[i] = np.array([float(match.group("YPOS")), float(match.group("XPOS"))])
    return output




def getSortIndecies(files, coords, maintain_read_order = True, keep_relative_coords=False):

    if not keep_relative_coords:
        # for i in range(len(files)):
        #     print(files[i]["index"])
        # metadata files does not give accurate data. 
        # for item in files:
        #     print(item["pos"] - files[0]["pos"])
        under_zero = coords[-1] < 0 


        if np.any(under_zero):
            c = np.where(under_zero, -1, 1)
            coords = coords*c
            
        
    if not maintain_read_order:
        if coords.shape[1] != 4:
            pass
            #Run through files
        print(coords)
        max_row = np.amax(coords[:, 0]) + 1
        max_col = np.amax(coords[:, 1]) + 1
        sorted_row = coords[np.argsort(coords[:, 0])]
        
        for i in range(max_row):
            sorted_row[i*max_col:(i+1)*max_col] = sorted_row[i*max_col:(i+1)*max_col][np.argsort(sorted_row[i*max_col:(i+1)*max_col, 1])]
        coords = sorted_row
    
    return coords






