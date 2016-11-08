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

def compute_ycrcb(R, G, B):
    y  = 16 + (int)( 0.257 * R + 0.504 * G + 0.098 * B)
    cb = 128 + (int)(-0.148 * R - 0.291 * G + 0.439 * B)
    cr = 128 + (int)( 0.439 * R - 0.368 * G - 0.071 * B)

    return y, cr, cb


def compute_hsv(R, G, B):
    R1, G1, B1 = float(R)/255.0, float(G)/255.0, float(B)/255.0
    Cmax = max(R1, G1, B1)
    Cmin = min(R1, G1, B1)
    delta = Cmax - Cmin

    if delta == 0:
        H = 0
    elif Cmax == R1:
        H = 60 * ((G1 - B1) / delta) % 6 
    elif Cmax == G1:
        H = 60 * ((B1 - R1) / delta + 2) 
    elif Cmax == B1:
        H = 60 * ((R1 - G1) / delta + 4) 
    
    if Cmax == 0:
        S = 0
    else:
        S = float(delta / Cmax)

    V = Cmax

    return H, S, V

def color_segmentation(img):
    res = deepcopy(img)
    M = len(res)
    N = len(res[0])

    for i in range(M):
        for j in range(N):
            [R, G, B] = img[i][j]
            H, S, V = compute_hsv(R, G, B)
            Y, Cr, Cb = compute_ycrcb(R, G, B)

            # Consider R, G, B values for human skin at
            # daylight or flashlight/daylight lateral
            r1 = R > 50 and G > 40 and B > 20 \
                and max(max(R, G), B) - min(min(R, G), B) > 10 \
                and R - G >= 10 and R > G and R > B

            r2 = R > 220 and G > 210 and B > 170 and R > B \
                and G > B and abs(R - G) <= 15

            rule_A = r1 or r2

            rule_B = H >= 0 and H <= 50 and S >= 0.1 and S <= 0.9

            r3 = Cb >= 60 and Cb <= 130
            r4 = Cr >= 130 and Cr <= 165

            rule_C = r3 and r4

            if rule_A and rule_B and rule_C:
                res[i][j] = [255, 255, 255]
            else:
                res[i][j] = [0, 0, 0]

    return res

def get_valid_neigh_pos(l, c, lmin, cmin, L, C):
    neigh_pos = [(i, j) for i in range(l-1, l+2) for j in range(c-1, c+2)]
    final_pos = []

    for (i, j) in neigh_pos:
        if i >= lmin and i < L and j >= cmin and j < C:
            final_pos.append((i, j))

    return final_pos


def find_connected(img):
    M = len(img)
    N = len(img[0])
    connected = np.zeros((M, N))
    regions = {}
    mark = 0

    for i in range(M):
        for j in range(N):
            if img[i][j][0] == 255 and connected[i][j] == 0:
                neigh_pos = get_valid_neigh_pos(i, j, 0, 0, M, N)
                marks = [connected[l][c] for (l, c) in neigh_pos if connected[l][c] != 0]

                if len(marks) > 0:
                    label = min(marks)
                    regions[label].append((i, j))
                    connected[i][j] = label
                else:
                    mark += 1
                    regions[mark] = [(i, j)]
                    connected[i][j] = mark
    return regions

def remove_little_pieces(img, regions):
    res = deepcopy(img)
    regs = deepcopy(regions)

    for label in regions:
        if len(regions[label]) < 40:
            for (i, j) in regions[label]:
                res[i][j] = [0, 0, 0]

            del regs[label]

    return res, regs

def length(region):
    y_ces = [j for (i, j) in region]
    ymin, ymax = min(y_ces), max(y_ces)
    return ymax - ymin 

def breadth(region):
    x_es = [i for (i, j) in region]
    xmin, xmax = min(x_es), max(x_es)
    return xmax - xmin

def ratio(region):
    l = length(region)
    b = breadth(region)
    return float(b)/float(l)

def eccen(region):
    l = length(region)
    b = breadth(region)

    # if b > l:
    #     return 0.0

    return math.sqrt(float(abs(b*b - l*l))) / float(b)

def refine_by_form(img, regions):
    res = deepcopy(img)
    regs = deepcopy(regions)

    for label in regions:
        r = ratio(regions[label])
        e = eccen(regions[label])
        print r, e
        if r < 0.4 or r > 1.1:# or e < 0.25 or e > 0.97:
            del regs[label]
            print "removed"
            for (i, j) in regions[label]:
                res[i][j] = [0, 0, 0]

    return res, regs

'''
old: old value of the pixel
new: new val of the px
'''
def remove_noise(res, m, n, M, N, old, newc, threshold):

    ok = True
    while ok == True:
        dbl = deepcopy(res)
        ok = False
        k = 1
        for i in range(m, M):
            for j in range(n, N):
                total = 0; count = 0
                [Ro, Go, Bo] = res[i][j]
                if Ro == old and Go == old and Bo == old:
                    # here, if instead of m, n, M, N, I let
                    # 0, 0, len(res), len(res[0]), I obtain the actual
                    # shape of the octagon in the binary image
                    for p in range(max(i - k, 0), min(len(res), i + k + 1)):
                        for q in range(max(j - k, 0), min(len(res[0]), j + k + 1)):
                            total += 1;
                            [R, G, B] = dbl[p][q]
                            if R == newc and G == newc and B == newc:
                                count += 1

                    if total - count < count + threshold:
                        ok = True
                        res[i][j] = [newc, newc, newc]

    return res

'''
First trial to determine the corners.
'''
def get_corners_trial(res):
    # Get white pixels set
    white = [(i, j) for i in range(len(res)) \
    for j in range(len(res[i])) if res[i][j][0] == 255]

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

    lmax = min(len(res) - 1, B[0] + 1)
    lmin = max(0, A[0] - 1)
    cmax = min(len(res[0]) - 1, C[1] + 1)
    cmin = max(0, A[1] - 1)

    return lmin, cmin, lmax, cmax


def main():
    #file_img_name = 'dk_trainers.jpg'
    #file_img_name = 'junior_it_club.jpg'
    #file_img_name = 'summer.jpg'
    #file_img_name = 'audrey_hpb.jpg'
    #file_img_name = 'edenland.jpg'
    #file_img_name = "prom.jpg"
    #file_img_name = "final.jpg"
    #file_img_name = 'breakout.jpg'
    #file_img_name = 'queen_2b.jpg'
    file_img_name = 'dr_house.jpg'

    f = misc.face()
    f = misc.imread(file_img_name)

    res = color_segmentation(f)

    regions = find_connected(res)

    res, regions = remove_little_pieces(res, regions)


    res = remove_noise(res, 0, 0, len(res), len(res[0]), 255, 0, 1)

    # # First trial of computing corners
    # lmin, cmin, lmax, cmax = get_corners_trial(res)

    # # Remove white parts with more blach than white around
    # res = remove_noise(res, lmin, cmin, lmax + 1, cmax + 1, 0, 255, 2)

    for label in regions:
        visited = deepcopy(res)
        lines = [i for (i, j) in regions[label]]
        cols = [j for (i, j) in regions[label]]
        lmin, lmax = min(lines), max(lines)
        cmin, cmax = min(cols), max(cols)

        for i in range(lmin, lmax):
            for j in range(cmin, cmax):
                if visited[i][j][0] == 0:
                    neigh_pos = get_valid_neigh_pos(i, j, lmin, cmin, lmax, cmax)
                    neigh_vals = [res[l][c][0] for (l, c) in neigh_pos if visited[l][c][0] == 255]
                    total = len(neigh_pos) - 1
                    white = len(neigh_vals)

                    if total - white <= white:
                        res[i][j] = [255, 255, 255]
                        regions[label].append((i, j))


        #res = remove_noise(res, lmin, cmin, lmax + 1, cmax + 1, 0, 255, 2)
    
    res, regions = refine_by_form(res, regions)
    # print len(regions2)
    # print "Regions 3: "
    # print len(regions3)

    # plt.imshow(f)
    # plt.show()

    plt.imshow(res)
    plt.show()
   

main()