import cv2 as cv
import numpy as np

"""
Most of this code is generated from chatGPT

"""



def generateRandomRectangle(N: int, M: int, scale: float, width_range: list=[200, 500], height_range: list = [200, 500]) -> (np.ndarray, np.ndarray):
    """
    Generates two rectangles with centrum in the middle of an NxM square. 
    The width and height of the smallest rectangle is by default between 100 and 500. 
    The larger rectangle is the smaller rectangle scaled by scale in size.

    N: Width of image
    M: height of image
    scale: scale between the rectangles
    width_range: 2 len list of [width_lower_bound, width_upper_bound]
    height_range: 2 len list of [height_lower_bound, height_upper_bound]
    
    returns: 
        smaller rectange: numpy array of coordinates for rectangle [x, y]
        larger rectangle: numpy array of coordinates for rectangle [x, y]
    
    """
    # Generate random width and height for the rectangle
    width = np.random.randint(width_range[0], width_range[1])
    height = np.random.randint(height_range[0], height_range[1])

    # print(width, height)

    # angle_degrees = 0
    angle_degrees = np.random.uniform(0, 360)
    # Generate rectangle vertices centered in the image
    center_x = M // 2
    center_y = N // 2

    half_width = width // 2
    half_height = height // 2

    rectangle = np.array([
        [center_x - half_width, center_y - half_height],
        [center_x + half_width, center_y - half_height],
        [center_x + half_width, center_y + half_height],
        [center_x - half_width, center_y + half_height]
    ])
    rectangle_2 = np.array([
        [center_x - half_width*scale, center_y - half_height*scale],
        [center_x + half_width*scale, center_y - half_height*scale],
        [center_x + half_width*scale, center_y + half_height*scale],
        [center_x - half_width*scale, center_y + half_height*scale]
    ])
    
    rotated_rectangle_1 = rotateRectangle(rectangle, angle_degrees)
    rotated_rectangle_2 = rotateRectangle(rectangle_2, angle_degrees)

    return rotated_rectangle_1, rotated_rectangle_2
   
   

def rotateRectangle(rectangle: np.ndarray, angle_degrees: int) -> np.ndarray:
    """
    Rotates the input rectangle with angle degrees, 

    Rectangle: np array with coordinates for corner of rectangle
    angle_degrees: angle in degrees of rotation

    return: the resulting rectangle
    
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Calculate the center of the rectangle
    center = np.mean(rectangle, axis=0)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Translate the rectangle to the origin, rotate, and translate back
    rotated_rectangle = np.dot(rectangle - center, rotation_matrix.T) + center

    return rotated_rectangle

def drawRectangle(image, rectangle, color, thickness):
    if type(color) is int:
        if len(image.shape) == 3:
            c = (color, color, color)
        else: 
            c = color
    else:
        if len(image.shape) == 2:
            c = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        else:
            c = color

    return cv.polylines(image, [rectangle.astype(int)], True, c, thickness)


def generateAndDrawRectangles(image: np.ndarray, scale: float=1.5, color: list or int=(0, 0, 0), thickness: int=2, for_hough=False, return_rectangles=False)-> np.ndarray:
    """
    Generates random rectangles with the paraemters given and draws them on top of image

    iamge: the image to draw the rectangle on
    scale: the scale betwene the two rectangles
    color: the color of the rectangle, can be grayscale or BGR
    thickness: thicness of the line that draws the rectangle
    for_hough: if true returns two images, one with the given thickness and one where thickness is equal to one
    
    returns
    image
    """
    
    
    rectangle_1, rectangle_2 = generateRandomRectangle(image.shape[0], image.shape[1], scale)
    
    if for_hough:
        image2 = image.copy()
        image2 = drawRectangle(image2, rectangle_1, color, thickness=1)
        image2 = drawRectangle(image2, rectangle_2, color, thickness=1)

    # Draw the rectangle on the image
    image = drawRectangle(image, rectangle_1, color, thickness=thickness)
    image = drawRectangle(image, rectangle_2, color, thickness=thickness)

    if for_hough:
        return image, image2
    if return_rectangles:
        return image, rectangle_1, rectangle_2
    return image

