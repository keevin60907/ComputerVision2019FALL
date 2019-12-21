'''
Computer Vision @ NTUEE 2019 Fall
Assignment 4 Stereo Matching

Author  : Kevin Yang
Date    : 2019/12/17 
'''
import time
import numpy as np
import cv2
from cv2.ximgproc import weightedMedianFilter, guidedFilter

def census_cost(mat_1, mat_2):
    '''
    The function is used to calculate the census cost with the type np.array((k, k), dtype=boolean)

    Arg(s):
        mat_1(np.array) : BGR window with shape (k, k , 3)
        mat_2(np.array) : BGR window with shape (k, k , 3)

    Return(s):
        cost(int)       : XOR result of the two windows
    '''
    radius = mat_1.shape[0]//2
    medium_1, medium_2 = mat_1[radius, radius], mat_2[radius, radius]
    # convert teh window into boolean form
    mat_1 = mat_1 >= medium_1
    mat_2 = mat_2 >= medium_2
    mat_1, mat_2 = mat_1.reshape(-1, 1), mat_2.reshape(-1, 1)
    cost = np.sum(np.equal(mat_1, mat_2) == False)
    return cost

def build_cost_volume(img_left, img_right, max_disp):
    '''
    Building cost volume in the traditional pipeline of stereo matching
    Brutal force calculation the cost of each pixel from two input images

    Arg(s):
        img_left(np.array)  : BGR image with left view field
        img_right(np.array) : BGR image with right view field
        max_dsip(int)       : the max disparity for calculation

    Return(s):
        disparity_left(np.array)    : cost volume of img_left with shape (height, width, max_disp)
        disparity_right(np.array)   : cost volume of img_right with shape (height, width, max_disp)
    '''
    height, width, _ = img_right.shape
    disparity_left = np.ones((height, width, max_disp)) * 255
    disparity_right = np.ones((height, width, max_disp)) * 255
    rad = 3
    padded_left = cv2.copyMakeBorder(img_left, rad, rad, rad, rad, cv2.BORDER_DEFAULT)
    padded_right = cv2.copyMakeBorder(img_right, rad, rad, rad, rad, cv2.BORDER_DEFAULT)
    for coor_y in range(height):
        print('processing...{:.2f}%\r'.format(coor_y/height*100), end='')
        for coor_x in range(width):
            for disp in range(max_disp):
                # slide the window for each pixel
                window_y = np.arange(coor_y, coor_y+2*rad+1)
                window_x = np.arange(coor_x, coor_x+2*rad+1)
                # make sure for the boarders
                if coor_x-disp < 0:
                    disparity_left[coor_y, coor_x, disp] = disparity_left[coor_y, coor_x, disp-1]
                else:
                    disparity_left[coor_y, coor_x, disp]\
                    = census_cost(padded_left[window_y][:, window_x],\
                                  padded_right[window_y][:, window_x-disp])
                if coor_x + disp >= width:
                    disparity_right[coor_y, coor_x, disp] = disparity_left[coor_y, coor_x, disp-1]
                else:
                    disparity_right[coor_y, coor_x, disp]\
                    = census_cost(padded_left[window_y][:, window_x+disp],\
                                  padded_right[window_y][:, window_x])
    return disparity_left, disparity_right

def consistency_check(disparity_left, disparity_right, max_disp):
    '''
    Follow the formula of the left-right checking
    Record the coordination of the holes

    Arg(s):
        disparity_left(np.array)    : the disparity map of img_left with shape (height, width)
        disparity_right(np.array)   : the disparity map of img_right with shape (height, width)
        max_dsip(int)               : the max disparity for calculation

    Return(s):
        ret(np.array)   : disparity map after left-rigth cheacking
        holes(set)      : the set of coordination of inconsistent points
    '''
    height, width = disparity_left.shape
    ret = disparity_left
    holes = set()
    for coor_y in range(height):
        for coor_x in range(max_disp, width):
            # the original check formula: Dl(x, y) == Dr(x-Dl(x, y), y)
            dis = int(disparity_left[coor_y, coor_x])
            if disparity_left[coor_y, coor_x] == disparity_right[coor_y, coor_x-dis]:
                ret[coor_y, coor_x] = disparity_left[coor_y, coor_x]
            else:
                holes.add((coor_y, coor_x))
    print('There are {:<03d} holes after left-right check'.format(len(holes)))
    return ret, holes

def hole_filling(disparity, holes):
    '''
    Follow the formula of the left-right checking
    Record the coordination of the holes

    Arg(s):
        disparity(np.array) : disparity map after left-rigth cheacking with shape (height, width)
        holes(set)          : the set of coordination of inconsistent points

    Return(s):
        disparity(np.array)   : disparity map after hole-filling with shape (height, width)
    '''
    width = disparity.shape[1]
    for (hole_y, hole_x) in holes:
        left_point, right_point = 0, 0
        for pixel in range(1, 21):
            # if we find the valid points, stop the loop
            if left_point != 0 and right_point != 0:
                break
            # hold the boarders while searching the right side
            if hole_x + pixel >= width:
                pass
            elif (hole_y, hole_x + pixel) not in holes:
                right_point = pixel
            # hold the boarders while searching the left side
            if hole_x - pixel < 0:
                pass
            elif (hole_y, hole_x - pixel) not in holes:
                left_point = pixel
        # get the nearest valid pixel to fill the hole
        if right_point < left_point:
            disparity[hole_y, hole_x] = disparity[hole_y, hole_x + right_point]
        else:
            disparity[hole_y, hole_x] = disparity[hole_y, hole_x - left_point]
    return disparity

def computeDisp(img_left, img_right, max_disp):
    '''
    The mainfunction of disparity computation

    Arg(s):
        img_left(np.array)  : BGR image with left view field
        img_right(np.array) : BGR image with right view field
        max_dsip(int)       : the max disparity for calculation

    Return(s):
        disparity(np.array) : the disparity map of final result (height, width, max_disp)
    '''
    img_left = img_left.astype(np.float32)
    img_right = img_right.astype(np.float32)

    # >>> Cost computation
    start_time = time.time()
    disparity_left, disparity_right = build_cost_volume(img_left, img_right, max_disp)
    end_time = time.time()
    print()
    print('time of build cost volume: {:.3f} s'.format(end_time - start_time))

    # >>> Cost aggregation
    start_time = time.time()
    for disp in range(max_disp):
        disparity_left[:, :, disp] = guidedFilter(img_left.astype(np.uint8),
                                                  disparity_left[:, :, disp].astype(np.uint8),
                                                  16, 50, -1)
        disparity_right[:, :, disp] = guidedFilter(img_right.astype(np.uint8),
                                                   disparity_right[:, :, disp].astype(np.uint8),
                                                   16, 50, -1)
    end_time = time.time()
    print('time of cost aggregation: {:.3f} s'.format(end_time - start_time))

    # >>> Disparity optimization (Winner Take All)
    start_time = time.time()
    disparity_left = np.argmin(disparity_left, axis=2).astype(np.uint8)
    disparity_right = np.argmin(disparity_right, axis=2).astype(np.uint8)
    end_time = time.time()
    print('time of disparity optimization: {:.3f} s'.format(end_time - start_time))

    # >>> Disparity refinement (Left-right consistency check)
    start_time = time.time()
    disparity, holes = consistency_check(disparity_left, disparity_right, max_disp)
    disparity = hole_filling(disparity, holes)
    disparity = weightedMedianFilter(img_left.astype(np.uint8),
                                     disparity.astype(np.uint8),
                                     23, 5, cv2.ximgproc.WMF_JAC)
    end_time = time.time()
    print('time of disparity refinement: {:.3f} s'.format(end_time - start_time))

    return disparity.astype(np.uint8)
