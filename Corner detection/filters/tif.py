import tifffile
import numpy as np


def read(filename, key=None):
    if key is None:
        return tifffile.imread(filename)
    else:
        return tifffile.imread(filename, key=key)

def readAndNorm(filename, key, maks=255):
    image = read(filename, key)
    return (image*(maks/np.amax(image))).astype(np.uint8)

def readMetadata(filename):
    return tifffile.TiffFile(filename)

def writeData(filename, data, photometric):
    tifffile.imwrite(filename, data, photometric)