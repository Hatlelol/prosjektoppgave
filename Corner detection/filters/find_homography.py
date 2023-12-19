#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: Sondre Hatlehol
MatrNr: 12202296
"""
from typing import Callable

import time

import numpy as np
import sklearn.cluster

import random

from helper_functions import *


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Boolean array with the same length as target_points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################
    # Write your own code here

    numUnderT = 0 # saving number of inliers under threshold value

    best_inliers = np.full(shape=len(target_points), fill_value=False, dtype=bool)

    k = 0
    while (1 - (best_inliers.sum()/len(target_points)) ** 4) ** k >= 1 - confidence:
        # Realization of the formula for ransac in the lectures
        k += 1
        usedPoints = []
        i = 0
        while i < 4: # Generate 4 unique random points
            p = random.randrange(0, len(source_points))
            if p not in usedPoints:
                i += 1
                usedPoints.append(p)

        s = np.array([source_points[m] for m in usedPoints])
        t = np.array([target_points[m] for m in usedPoints])
        H = find_homography_leastsquares(s, t)

        points = np.c_[source_points, np.ones(len(source_points))] # Add a column of ones to correspond with the dimensions of H

        proj = (H @ points.T).T


        proj = (proj[..., 0:2].T / proj[..., 2]).T


        #proj = (H @ points.T).T # Gotten from draw_rectangles()

        # As distance between two points a and b is ((a1-b1)^2 + (a2-b2)^2)^1/2 then the Frobenius  norm
        # of a-b will be the distance between the points. np.linalg.norm has Frobenius norm by default:
        #d = np.linalg.norm(proj - np.c_[target_points, np.ones(len(target_points))], axis=1)
        d = np.linalg.norm(proj - target_points, axis=1)

        dBool = np.where(d < inlier_threshold, True, False)
        s = dBool.sum()
        if s > numUnderT:
            numUnderT = s
            bestH = H
            best_inliers = dBool
            #best_inliers = np.logical_or(dBool, best_inliers) # This code line is what I imagined from the exersice text, but the current
                                                               # implementation works better


    source = source_points[best_inliers]
    target = target_points[best_inliers]

    H2 = find_homography_leastsquares(source, target)

    best_suggested_homography = H2
    num_iterations = k

    #print(f"Number of iterations: {k}")
    ######################################################
    return best_suggested_homography, best_inliers, num_iterations


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    # Write your own code here

    if len(source_points) < 4 or len(target_points) < 4:
        print("Not enough mathing points!")
        return np.eye(3)


    matrix = np.zeros(shape=(len(source_points)*2, 8))

    i = 0
    for s, t in zip(source_points, target_points): # Create the rows of the matrix x in the eq xH=p as defined in exercise sheet.
        # zip funcition discards the latter parts of the longer array if the arrays are not equal in length.
        matrix[i]     = np.array([s[0], s[1], 1, 0, 0, 0, -t[0]*s[0], -t[0]*s[1]])
        matrix[i + 1] = np.array([0, 0, 0, s[0], s[1], 1, -t[1]*s[0], -t[1]*s[1]])
        i += 2


    x = np.array(target_points.flatten()) # get p array

    solution = np.linalg.lstsq(matrix, x)

    homography = solution[0]

    ######################################################
    return np.reshape(np.append(homography, 1), (-1, 3)) # Add back one to H and reshape it to 3x3
