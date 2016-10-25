import numpy as np

def getHammingWin(n):
    ham=[0.54-0.46*np.cos(2*np.pi*i/(n-1)) for i in range(n)]
    ham=np.asanyarray(ham)
    ham/=ham.sum()
    return ham

def myLpf(data, hamming):

    N=len(hamming)
    res=[]
    for n, v in enumerate(data):
        y=0
        for i in range(N):
            if n<i:
                break
            y+=hamming[i]*data[n-i]
        res.append(y)
    return np.asanyarray(res)