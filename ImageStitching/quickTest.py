import numpy as np
import os
import re
import shutil
ones1 = np.ones((10, 10))
ones2 = np.ones((8, 9))



def checkAndPadArrays(arr1: np.ndarray, arr2: np.ndarray) -> (np.ndarray, np.ndarray):

    shape1 = arr1.shape
    shape2 = arr2.shape

    if shape1[0] > shape2[0]:
        arr2 = np.pad(arr2, [(0, shape1[0] - shape2[0]), (0, 0)])
    elif shape1[0] < shape2[0]:
        arr1 = np.pad(arr1, [(0, -shape1[0] + shape2[0]), (0, 0)])



    if shape1[1] > shape2[1]:
        arr2 = np.pad(arr2, [(0, 0), (0, shape1[1] - shape2[1])])
    elif shape1[1] < shape2[1]:
        arr1 = np.pad(arr1, [(0, 0), (0,  -shape1[1] + shape2[1])])

    return arr1, arr2

# ones1, ones2 = checkAndPadArrays(ones1, ones2)
# print(ones1)
# print(ones2.shape)
# print(ones2)
# i = 0
# folder = R"C:\Users\sondr\Downloads\dsd_10\dsd_10"
# for file_path in os.listdir(folder):
#     if os.path.isfile(os.path.join(folder, file_path)):
#         if file_path[-3:] == "txt":
#             with open(os.path.join(folder, file_path)) as f:
#                 file = [line.strip() for line in f.readlines()]
#             if i == 0:
#                 prevfile = file
#             else:
#                 eq_l = 0
#                 for k in range(min(len(file), len(prevfile))):
                    
#                     if file[k] != prevfile[k]:
#                         eq_l += 1
#                         print(f"{k}: {file[k]}   {prevfile[k]}")
#                 print(f"Files have {len(file) - eq_l} equal lines and {eq_l} noneq")
#                 print()
#                 prevfile = file
#             i += 1

def getTileConfig(file):
    with open(file) as f:
        data_config = f.read()
    r = r'\(([-\d\.]+), ([-\d\.]+)\)'
    s = re.findall(r, data_config)
    output = []
    for m in s:
        output.append((float(m[0]), float(m[1])))
    return output


with open(R"C:\data\dsd_10\dsd_10\dsd_10_MMStack_1-Pos004_010_metadata.txt") as f:
    data = f.read()


r = r'"Label": "([\w-]+)",.*?"GridRow": (\d+).+?"GridCol": (\d+).+?"Position_um": \[\n +([\d\-.]+),\n +([\d\-.]+)'
s = re.findall(r, data, flags=re.DOTALL)

first = True
x_offset = 0
y_offset = 0

files = []



for m in s:
    
    Label = m[0]
    Row = int(m[1])
    Col = int(m[2])
    x_pos = float(m[3])
    y_pos = float(m[4])
    if first:
        first = False
    
    curr_index = (Row, Col)
    pos =  re.search(r'1-Pos(\d{3})_(\d{3})', Label)
    if Row != int(pos.group(2)):
        print(f"Wrong row for label {Label}, {Row} {pos.group(2)}")
    if Col != int(pos.group(1)):
        print(f"Wrong col for label {Label}, {Col} {pos.group(1)}")

    obj = {
        "label": Label,
        "index": curr_index,
        "pos": (y_pos, x_pos)
    }
    files.append(obj)

tileconfig = R"C:\data\dsd_10\dsd_10\TileConfiguration.registered.txt"
tilecoords = getTileConfig(tileconfig)


for i in range(1, int(len(files)/5), 2):
    
    files[i*5:(i+1)*5] = files[i*5:(i+1)*5][::-1]
    tilecoords[i*5:(i+1)*5] = tilecoords[i*5:(i+1)*5][::-1]

nparrsave = np.array(tilecoords)
nparrsave = nparrsave.reshape(int(len(files)/5), 5, 2)
print(nparrsave)
with open("coords.npy", "wb") as f:
    np.save(f, nparrsave)





def reorder(f, c):
    reordered = []
    c_reordered = []
    # for i in range(int(len(files)/5), 0, -1):
    #     reordered.extend(files[len(files)-((i+1)*5):len(files)-(i*5)])
    for i in range(0, int(len(f)/5)):
        reordered.extend(f[len(f)-((i+1)*5):len(f)-(i*5)])
        c_reordered.extend(c[len(c)-((i+1)*5):len(c)-(i*5)])

    topLeft = reordered[0]
    topLeftPos = c_reordered[0]
    print(topLeftPos)
    for file in reordered:
        # print(f"{file['label']}: \t {file['pos']}\t{(file['pos'][0], abs(file['pos'][1] - topLeft['pos'][1]))}")
        # file["pos"] = (file["pos"][0], abs(file["pos"][1] - topLeft["pos"][1]))
        print(f"{file['label']}: \t {file['pos']}\t{(file['pos'][0] - topLeftPos[0], file['pos'][1] - topLeftPos[1])}")
        file["pos"] = (file["pos"][0] - topLeftPos[0], file["pos"][1] - topLeftPos[1])
        nums = re.search(r'1-Pos(\d{3})_(\d{3})', file['label'])
        toFile = f"C:\data\dsd_10\dsd_10\Reordered\dsd_10_MMStack_1-Pos{nums.group(1)}_{abs(int(nums.group(2)) - 14)}.ome.tif"
        shutil.copy(f"C:\data\dsd_10\dsd_10\dsd_10_MMStack_{file['label']}.ome.tif", toFile)

    return reordered

def reverse_reorder(f, c):
    reordered = []
    while len(reordered) != len(f):
        for i, file in enumerate(f):
            if len(reordered) == 0:
                if file['index'] == (14, 0):
                    file['pos'] = c[i]
                    print(c[i])
                    reordered.append(file)
                    
                continue 

            if reordered[-1]['index'][1] == 4:
                if file['index'] == (reordered[-1]['index'][0] - 1, 0):
                    file['pos'] = c[i]
                    print(c[i])
                    reordered.append(file)
            else:
                if file['index'] == (reordered[-1]['index'][0], reordered[-1]['index'][1] + 1):
                    file['pos'] = c[i]
                    print(c[i])
                    reordered.append(file)
    return reordered


def convertOrder(files):
    firstFile = files[0]
    originCoords = firstFile['pos']
    output = []
    for file in files:
        output.append({
        "label": f"1-Pos{file['index'][1] :03d}_{abs(file['index'][0] - 14) :03d}",
        "index": (abs(file['index'][0] - 14), file['index'][1]),
        "pos": (file['pos'][0] - firstFile['pos'][0], file['pos'][1] - firstFile['pos'][1])
    })
    return output


# files = reverse_reorder(files, tilecoords)
files_converted = convertOrder(files)


for i in range(len(files)):
    toFile = f"C:\data\dsd_10\dsd_10\Reordered\dsd_10_MMStack_{files_converted[i]['label']}.ome.tif"
    shutil.copy(f"C:\data\dsd_10\dsd_10\dsd_10_MMStack_{files[i]['label']}.ome.tif", toFile)



with open("C:\data\dsd_10\dsd_10\Reordered\out.txt", 'a') as f:
    for file in files_converted:
        f.write(f"dsd_10_MMStack_{file['label']}.ome.tif;;({file['pos'][0]}, {file['pos'][1]})\n")

for file in files_converted:
    print(f"dsd_10_MMStack_{file['label']}.ome.tif;;({file['pos'][0]}, {file['pos'][1]})")
