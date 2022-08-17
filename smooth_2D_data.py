import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib_style

def smooth_data(r, gr, Gaussian=True, sigmaliss=0.05, RunningAverage=False, 
                movingwidth=4, lowess=False, lwidth=0.02, 
                showplot=False, printsmoothed=False):

    #r  = list()
    #gr = list()

    if Gaussian:
        dim_to_liss = len(r) - 1
        fact_liss = 1 / ( sigmaliss * np.sqrt(2 * np.pi) )
        smoothedR = list()

        for INDA in range(dim_to_liss):
            xx = 0.0
            RRR = []
            for INDB in range(dim_to_liss):
                if INDB == 0:
                    DQ = r[1] - r[0]
                elif INDB == dim_to_liss:
                    DQ = r[dim_to_liss] - r[dim_to_liss-1]
                else:
                    DQ = 0.5 * (r[INDB+1] - r[INDB-1])
                xx = np.exp(-(r[INDB] - r[INDA])**2 / (2 * sigmaliss**2)) * gr[INDB] * DQ
                RRR.append(xx)
            smoothedR.append(sum(RRR) * fact_liss)
    
    return r[:dim_to_liss-1], smoothedR[:dim_to_liss-1]

    # if showplot:
        # plt.plot(r[:dim_to_liss-1], smoothedR[:dim_to_liss-1], label='Gaussian smoothening')

    #if printsmoothed:

    #Method 2: Moving average box (by convolution)
    # if RunningAverage:
    #     def smooth(y, box_pts):
    #         box = np.ones(box_pts)/box_pts
    #         y_smooth = np.convolve(y, box, mode='same')
    #         return y_smooth
    #     SmoothedGR = list()
    #     SmoothedGR = smooth(gr, movingwidth)
    #     if showplot:
    #         plt.plot(r, SmoothedGR, label='Moving average box')
    #     # if printsmoothed:
    #     #     print(' ', file=fileout)
    #     #     print('#Moving average box ', file=fileout)
    #     #     print('# r(ang) gr', file=fileout)
    #     #     for cx, ccx in list(zip(np.array(r),np.array(SmoothedGR))):
    #     #         print(cx,ccx, file=fileout)

    # #Method 3: Lowess similar to moving average
    # if lowess:
    #     lowess = sm.nonparametric.lowess(gr, r, frac=Lwidth)
    #     if showplot:
    #         plt.plot(lowess[:, 0], lowess[:, 1], label='Lowess Smoothing')
    #     # if printsmoothed:
    #     #     print(' ', file=fileout)
    #     #     print('#Lowess Smoothing ', file=fileout)
    #     #     print('# r(ang) gr', file=fileout)
    #     #     for cxx, ccxx in list(zip(lowess[:, 0], lowess[:, 1])):
    #     #         print(cxx,ccxx, file=fileout)
