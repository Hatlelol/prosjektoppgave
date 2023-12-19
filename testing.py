import numpy as np


# matrix = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
# print(matrix)

# print(np.sum(matrix, axis=0))

def calculateDivision(slices, length, overlap):
    # if (length/slices).is_integer():
    #     return np.full(slices, int(lenght/slices))
    

    

    base_value = np.floor((length*(1-2/100))/(slices*(1-overlap/100)))
    # print(length/(slices*(1-overlap/100)))
    # print(base_value)
    # print(int(length / slices)/(1 + 1*overlap/100))
    # print(int(length / slices))

    # Create an array of base values

    base_arr = np.array([(base_value*(1-overlap/100))*i for i in range(slices)]) + base_value/2

    # Distribute the remainder evenly among the elements
    

    return base_arr, base_value


# arr, base_val = calculateDivision(10, 1000, 10)

# for i in range(len(arr)):
#     if i == 0:
#         continue
#     print((arr[i - 1] + base_val/2 - (arr[i] - base_val/2))/base_val)

# print(arr[0] - base_val/2)
# print(arr[-1] + base_val/2)
# print(10*111*(1-10/100))
# print(111*(1-10/100))


def slice(length, slices, overlap, error=None):
    increment = int(length/(slices + overlap/100))
    if error is not None:
        max_err_inc = slices*increment*error/100
        length -= max_err_inc
        increment = int/length/(slices + overlap/100)


    wHalf = int((increment + increment*overlap/100)/2)
    inc = increment
    for i in range(10):
        mid = wHalf + i*increment
        print(f"mid: {mid}, l: {mid - wHalf} r: {mid + wHalf} w: {mid + wHalf - (mid - wHalf)}")

    return wHalf, increment

    

baseVal = int(1000/((10+0.2)))
print(baseVal)
print(baseVal*10 + baseVal*0.2)


val = int(baseVal/2)
for i in range(10):
    print(baseVal*(i+1) + 2*baseVal*0.1)
    print(val*(i+1) - (val))


slice(1000, 10, 100)

