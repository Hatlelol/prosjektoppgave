import os

# 1. --filenamePattern <string containing filename pattern>
filenamePattern = R"img_r{rrr}_c{ccc}.jpg"
# 2. --filenamePatternType <ROWCOL/SEQUENTIAL>
filenamePatternType = "ROWCOL"
# 3a. --gridOrigin <UL/UR/LL/LR>  -- Required only for ROWCOL or SEQUENTIAL
gridOrigin = "UL"
# 3b. --gridDirection <VERTICALCOMBING/VERTICALCONTINUOUS/HORIZONTALCOMBING/HORIZONTALCONTINUOUS> -- Required only for SEQUENTIAL
gridDirection = ""
# 4. --gridHeight <#>
gridHeight = 2
# 5. --gridWidth <#>
gridWidth = 2
# 6. --imageDir <PathToImageDir>
#imageDir = R"C:\Users\sondr\Downloads\Small_Fluorescent_Test_Dataset\Small_Fluorescent_Test_Dataset\image-tiles"
imageDir = R"C:\Users\sondr\OneDrive\Dokumenter\a\Prosjektoppgave\Testing\MIST_test"
# 7a. --startCol <#> -- Required only for ROWCOL
startCol = 0
# 7b. --startRow <#> -- Required only for ROWCOL
startRow = 0
# 7c. --startTile <R> -- Required only for SEQUENTIAL
startTile = 0
# 8. --programType <AUTO/JAVA/FFTW> -- Highly recommend using FFTW
programType = "FFTW"
# 9. --fftwLibraryFilename libfftw3f.dll -- Required for FFTW program type
fftwLibraryFilename = "libfftw3f-3.dll"
# 9a. --fftwLibraryName libfftw3f -- Required for FFTW program type
fftwLibraryName = "libfftw3f-3"
# 9b. --fftwLibraryPath <path/to/library> -- Required for FFTW program type
# fftwLibraryPath = R"C:\Users\sondr\Downloads\fiji-win64\Fiji.app\lib\fftw\*"
# fftwLibraryPath = R'"C:\Users\sondr\Downloads\fiji-win64\Fiji.app\lib\fftw"'
fftwLibraryPath = R'"C:\Users\sondr\Downloads\fftw-3.3.5-dll64"'

outputPath = os.getcwd()

displayStitching = "true"

outputFullImage = "true"

# executrion:
# java.exe -jar MIST_-2.1-jar-with-dependencies.jar --filenamePattern img_r{rrr}_c{ccc}.tif --filenamePatternType ROWCOL --gridHeight 5 --gridWidth 5 --gridOrigin UR --imageDir C:\Users\user\Downloads\Small_Fluorescent_Test_Dataset\image-tiles --startCol 1 --startRow 1 --programType FFTW --fftwLibraryFilename libfftw3f.dll --fftwLibraryName libfftw3f --fftwLibraryPath C:\Users\user\apps\Fiji.app\lib\fftw

def checkGridDirection():
    if filenamePatternType == "SEQUENTIAL":
        return f"--gridDirection {gridDirection}"
    else:
        return ""
    
def checkStart():
    if filenamePatternType == "ROWCOL":
        return f"--startCol {startCol} --startRow {startRow} "
    elif filenamePatternType == "SEQUENTIAL":
        return f"--startTile {startTile} "
    else:
        return ""

def checkFFTW():
    if programType == "FFTW":
        return f"--fftwLibraryFilename {fftwLibraryFilename} \
--fftwLibraryName {fftwLibraryName} \
--fftwLibraryPath {fftwLibraryPath}"
    else:
        return ""

cm1 = f"--filenamePattern {filenamePattern} "
cm2 = f"--filenamePatternType {filenamePatternType} "
cm3 = f"--gridOrigin {gridOrigin} {checkGridDirection()} "
cm4 = f"--gridHeight {gridHeight} "
cm5 = f"--gridWidth {gridWidth} "
cm6 = f"--imageDir {imageDir} "
cm7 = checkStart()
cm8 = f"--programType {programType} {checkFFTW()} "

cm9 = f"--outputPath {outputPath} "
cm10 = f"--displayStitching {displayStitching} "
cm11 = f"--outputFullImage true "
commands = cm1 + cm2 + cm3 + cm4 + cm5 + cm6 + cm7 + cm8 + cm9 + cm10 + cm11



cmd = R"java -jar C:\MIST_Test\MIST\target\MIST_-2.1-jar-with-dependencies.jar " + commands
print(cmd)

returned_value = os.system(cmd)  # returns the exit code in unix
print('returned value:', returned_value)