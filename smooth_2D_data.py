import os
import numpy as np
import matplotlib.pyplot as plt
#import statsmodels.api as sm
#from statsmodels.nonparametric.kernel_regression import KernelReg
#import matplotlib_style

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


def weighted_linear_least_square (x: list,y: list ,weights: list) :
    ''' This performs the linear regression using weighted_linear_square.
        Compute the weighted means and weighted deviations from the means.
        wm denotes a "weighted mean", wm(f) = (sum_i w_i f_i) / (sum_i w_i)

    Paramater:
    --------------------------------------------------------------------
    x : Independent variable
    y : Dependent variable
    weights : weights of the data (w_i)

    f = \sum_i (w_i ( m' x_i + c')

    Here functions are minimized with weights w_i


    IMPORTANT : this function accepts inverse variances, not the 
                inverse standard deviations as the weights for the 
                data points.

    Result:
    --------------------------------------------------------------------
    c : Intercept of linear fitted line
    m : Slope of line fitted line
    cov_ij : Elements of covariance matrix
    chi2 : weighted sum of residuals
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    weights = np.asarray(weights)

    W = np.sum(weights)

    wm_x = np.average(x, weights=weights)
    wm_y = np.average(y, weights=weights)

    dx = x - wm_x
    dy = y - wm_y

    wm_dx2 = np.average(dx**2,weights=weights)
    wm_dxdy = np.average(dx*dy,weights=weights)

    # In terms of y = m x + c
    m = wm_dxdy / wm_dx2
    c = wm_y - wm_x*m

    cov_00 = (1.0/W) * (1.0 + wm_x**2/wm_dx2)
    cov_11 = 1.0 / (W*wm_dx2)
    cov_01 = -wm_x / (W*wm_dx2)
    
    # Compute chi^2 = \sum w_i (y_i - (a + b * x_i))^2
    chi2 = np.sum (weights * (y-(m*x+c))**2)

    
    return m, c, cov_00, cov_11, cov_01, chi2

