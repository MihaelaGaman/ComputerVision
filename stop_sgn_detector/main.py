from scipy import ndimage
from scipy import misc
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math
import colorsys
from copy import deepcopy
import cv2
from skimage import morphology


def main():
    file_img_name = 'ukni.jpg'
    f = misc.face()
    f = misc.imread(file_img_name)
    gamma = 2.2

    # Gamma correction
    ic = np.power(f, gamma)

    red = np.uint8([[[255,0,0 ]]])
    hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
    print hsv_red

    # Convert BGR to HSV
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([hsv_red[0][0][0] - 10,100,100])
    upper_blue = np.array([hsv_red[0][0][0] + 10,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(f,f, mask= mask)

    r = np.zeros((f.shape[0], f.shape[1]))

    # binary image
    for i in range(len(res)):
        for j in range(len(res[i])):
            if res[i][j][0] > 50:
                res[i][j][0] = 255
                res[i][j][1] = 255
                res[i][j][2] = 255
                r[i][j] = 255
                #print "i = ", i, " j = ", j
            else:
                res[i][j][0], res[i][j][1], res[i][j][2] = 0, 0, 0
    
    # remove noise
    ok = True
    while ok == True:
        dbl = deepcopy(res)
        ok = False
        k = 1
        M = len(res)
        for i in range(len(res)):
            N = len(res[i])
            for j in range(N):
                total = 0
                count = 0
                if res[i][j][0] == 255 and res[i][j][1] == 255 and res[i][j][2] == 255:
                    for m in range(max(i - k, 0), min(M, i + k+1)):
                        for n in range(max(j - k, 0), min(N, j + k+1)):
                            total += 1;
                            [R, G, B] = dbl[m][n]
                            if R == 0 and G == 0 and B == 0:
                                count += 1

                    #print "count = ", count, " total = ", total
                    if total - count < count + 1:
                        ok = True
                        res[i][j] = [0, 0, 0]
                    #    print i, j

    # Get white pixels set
    white = [(i, j) for i in range(len(res)) for j in range(len(res[i])) if res[i][j][0] == 255]
    white_l = [p[0] for p in white]
    white_c = [p[1] for p in white]

    #max_l, max_c, min_l, min_c
    il_min = white_l.index(min(white_l))
    il_max = white_l.index(max(white_l))
    ic_min = white_c.index(min(white_c))
    ic_max = white_c.index(max(white_c))
    
    A = (white_l[il_min], white_c[ic_min])
    B = (white_l[il_max], white_c[ic_min])
    C = (white_l[il_min], white_c[ic_max])
    D = (white_l[il_max], white_c[ic_max])

    up_corners = []

    # for l in range(len(res)):
    #     for c in range(len(res[i])):    

    print A, B, C, D
    lmax = min(len(f) - 1, B[0] + 1)
    lmin = max(0, A[0] - 1)
    cmax = min(len(f[0]) - 1, C[1] + 1)
    cmin = max(0, A[1] - 1)

    #f[A[0] : B[0]][A[1]] = [0, 255, 50]
    for i in range(A[0], B[0] + 1):
        # Vertical left line
        f[i][cmin] = [0, 0, 0]
        #f[i][A[1] - 2] = [0, 0, 0]

        # Vertical right
        f[i][cmax] = [0, 0, 0]
        #f[i][C[1] + 2] = [0, 0, 0]

    for j in range(A[1], C[1] + 1):
        # Horizontal up
        f[lmin][j] = [0, 0, 0]
        #f[A[0] - 2][j] = [0, 0, 0]

        # Horizontal down
        f[lmax][j] = [0, 0, 0]
        #f[B[0] + 2][j] = [0, 0, 0]

       # remove noise
    ok = True
    while ok == True:
        dbl = deepcopy(res)
        ok = False
        k = 1
        M = lmax + 1
        for i in range(lmin, M):
            N = cmax + 1
            for j in range(cmin, N):
                total = 0
                count = 0
                if res[i][j][0] == 0 and res[i][j][1] == 0 and res[i][j][2] == 0:
                    for m in range(max(i - k, lmin), min(M, i + k+1)):
                        for n in range(max(j - k, cmin), min(N, j + k+1)):
                            total += 1;
                            [R, G, B] = dbl[m][n]
                            if R == 255 and G == 255 and B == 255:
                                count += 1

                    #print "count = ", count, " total = ", total
                    if total - count < count + 2:
                        ok = True
                        res[i][j] = [255, 255, 255]
                        print i, j

    # Sablon colt stanga sus
    ul_mask = np.zeros(shape=(3,3))
    ul_mask[0][0] = 255; ul_mask[0][1] = 255
    ul_mask[0][2] = 255; ul_mask[1][0] = 255; ul_mask[2][0] = 255

    # Sablon colt stanga jos
    dl_mask = np.zeros(shape=(3,3))
    dl_mask[0][0] = 255; dl_mask[1][0] = 255
    dl_mask[2][0] = 255; dl_mask[2][1] = 255; dl_mask[2][2] = 255

    # Sablon colt dreapta sus
    ur_mask = np.zeros(shape=(3,3))
    ur_mask[0][0] = 255; ur_mask[0][1] = 255
    ur_mask[0][2] = 255; ur_mask[1][2] = 255; ur_mask[2][2] = 255

    # Sablon colt dreapta jos
    dr_mask = np.zeros(shape=(3,3))
    dr_mask[0][2] = 255; dr_mask[1][2] = 255
    dr_mask[2][2] = 255; dr_mask[2][1] = 255; dr_mask[2][0] = 255


    print ul_mask
    print dl_mask
    print ur_mask
    print dr_mask


    plt.imshow(f)
    plt.show()

    plt.imshow(res)
    plt.show()

    # plt.imshow(r, cmap = cm.Greys_r)
    # plt.show()
    

main()