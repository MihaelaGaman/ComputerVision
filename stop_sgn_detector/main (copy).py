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

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def rgb_to_hsv(img):
    img2 = deepcopy(img)
    for r in range(len(img)):
        for c in range(len(img[r])):
            [R, G, B] = img[r][c]

            # above = (0.5 * (R - G) + (R - B))
            # if above == 1:
            #     above = 1.01

            # below = (math.sqrt((math.pow(R - G, 2)) + (R - B) * (G - B)))
            # if below == 0:
            #     below = 0.01

            # H = (math.pow((math.cos((above / below))), -1))
            # S = ((3 / (R + G + B))) * min(R, G, B) / 100
            # V = (max(R, G, B) / 100)

            # H, S, V = rgb2hsv(R, G, B)
            H, S, V =  colorsys.rgb_to_hsv(R, G, B)

            img2[r][c][0], img2[r][c][1],img2[r][c][2] = H, S, V

    return img2

def main():
    file_img_name = 'stop-sign-pink-sherbet.jpg'
    f = misc.face()
    f = misc.imread(file_img_name)
    gamma = 2.2

    # Gamma correction
    ic = np.power(f, gamma)

    # Convert R,G,B to H, S, V
    #ihsv = rgb_to_hsv(f)
    #ihsv = 

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
                print "i = ", i, " j = ", j
            else:
                res[i][j][0], res[i][j][1], res[i][j][2] = 0, 0, 0
    
    a = np.zeros((2,2), dtype=np.int)
    a[1][1] = 1
    r = signal.convolve2d(r, a)

    # remove noise
    ok = True
    x = 0
    k = 0
    dbl = deepcopy(res)
    while ok == True:
        dbl = deepcopy(res)
        ok = False
        k = 1
        x += 1
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

                    print "count = ", count, " total = ", total
                    if total - count <= count:
                        ok = True
                        res[i][j] = [0, 0, 0]
                        print i, j

    plt.imshow(res)
    plt.show()

    # plt.imshow(r, cmap = cm.Greys_r)
    # plt.show()
    

main()