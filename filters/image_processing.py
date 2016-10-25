from scipy import ndimage
from scipy import misc
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gaussian_filter import matlab_style_gauss2D
from linear_filter import getHammingWin, \
                            myLpf     

def main():
    file_img_name = 'stop2.jpg'
    f = misc.face()
    f = misc.imread(file_img_name)

    gkernel = matlab_style_gauss2D()
    print gkernel

    hamming = getHammingWin(3)
    data = np.full((3, 3), -(float(1/9)))
    data[1][1] = -(float(17/9))
    print hamming
    print data

    grey = np.zeros((f.shape[0], f.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(len(f)):
        for colnum in range(len(f[rownum])):
            grey[rownum][colnum] = np.average(f[rownum][colnum])

    lkernel = myLpf(data, hamming)
    print lkernel

    grad = signal.convolve2d(grey, gkernel)
    lconv = signal.convolve2d(grey, lkernel)

    # plt.imshow(grey, cmap = cm.Greys_r)
    # plt.show()

    # plt.imshow(grad, cmap = cm.Greys_r)
    # plt.show()

    plt.imshow(lconv, cmap = cm.Greys_r)
    plt.show()


    # plt.imshow(grey)
    # plt.show()
    

main()