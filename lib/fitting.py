"""Various fitting functions for the DanMAX scripts"""

import numpy as np
import scipy.optimize as sci_op
import datetime
import multiprocessing as mp

def gauss(x,a,x0,sigma,bgr=0):
    """
    Return a gaussian peak function
    Parameters
        x     - x-values 
        a     - amplitude
        x0    - peak position in x-coordinates
        sigma - standard deviation (sigma = fwhm/(2*np.sqrt(2*np.log(2))))
        bgr   - (optional) constant background
    Return
        y     - y-values for the gaussian peak
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+bgr
    
def gaussCircle(chi,a,chi0,sigma,bgr=0., multiplicity=2):
    """
    Return a gaussian peak function on a circle
    Parameters
        chi          - chi-values 
        a            - amplitude
        chi0         - peak position in chi-coordinates
        sigma        - standard deviation (sigma = fwhm/(2*np.sqrt(2*np.log(2))))
        bgr          - (optional) constant background
        multiplicity - (optional) how many times does the peak repeat across the circle
    Return
        y_calc       - y-values for the gaussian peak
    """
    y_calc = np.zeros(chi.shape) 
    chi_shift = 2*np.pi / multiplicity  
    for i in range(multiplicity):
        y_calc += a * np.exp( - np.arccos(np.cos( chi-(chi0+chi_shift*i)))**2 / (2*sigma**2))
    y_calc += bgr

    return y_calc 


def gaussIntegral(a,sigma):
    """Return the integrated area of a gauss function"""
    return np.sqrt(2*np.pi)*a*np.abs(sigma)

def integralBreadth(x,y,bgr=0):
    """Return the numerical integral breadth"""
    I = np.abs(np.trapz(y-bgr,x))
    return I/np.max(y)

def numericalFWHM(x,y,bgr=None):
    """Estimate the numerical FWHM, assuming a gaussian peak shape"""
    if bgr is None:
        bgr = np.nanmin(y)
    beta = integralBreadth(x,y,bgr=bgr)
    return beta2FWHM(beta)
    
def beta2FWHM(beta):
    """Convert integral breadth (beta) to gaussian FWHM"""
    return beta*2*np.sqrt(np.log(2)/np.pi)

def sigma2FWHM(sigma):
    """Convert gaussian standard deviation to FWHM"""
    return sigma*(2*np.sqrt(2*np.log(2)))

def FWHM2sigma(fwhm):
    """Convert gaussian FWHM to variance"""
    return fwhm/(2*np.sqrt(2*np.log(2)))

def linearBackground(y,n=3):
    """
    Simple linear background, interpolated from the mean of the n first and n last points
    Parameters
        y   - y-data 1D or 2D array
        n   - (default=3) number of end-points to use for the mean
    Return
        bgr - background data - same shape as y
    """
    is_1D = len(y.shape)==1
    y = np.atleast_2d(y)
    bgr = np.linspace(np.mean(y[:,:n],axis=1),
                      np.mean(y[:,-n:],axis=1),
                      y.shape[-1])
    if is_1D:
        return bgr.T[0]
    return bgr.T


def singlePeakFit(x,y,verbose=True):
    """
    Performe a single peak gaussian fit
    Return: amplitude, position, FWHM, background, y_calc
    """
    # Peak profile - in this case gaussian
    def gauss(x,amp,x0,fwhm,bgr):
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        return amp*np.exp(-(x-x0)**2/(2*sigma**2))+bgr

    # initial position guess
    pos = x[np.argmax(y)]

    # initial background guess
    bgr = (y[0]+y[-1])/2
    
    # initial amplitude guess
    amp = np.nanmax(y)-bgr
    
    # Guess for sigma based on estimate of FWHM from array
    fwhm = np.abs(pos-x[np.argmin(np.abs(y-(amp/2)))])
    
    # Assume convergence before fit
    convergence = True

    # Fit the peak
    try:
        popt,pcov = sci_op.curve_fit(gauss,
                                     x,
                                     y,
                                     p0=[amp,pos,fwhm,bgr])
    except RuntimeError:
        if verbose:
            print('sum(X) fit did not converge!')
        convergence = False
        popt = [np.nan]*4

    amp,pos,fwhm,bgr = popt
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    integral = np.sqrt(2*np.pi)*amp*np.abs(sigma)
    
    y_calc = gauss(x,amp,pos,fwhm,bgr)
    
    return amp,pos,fwhm,bgr,y_calc


def multiGauss(x,peaks, bgr_glob=0.):
    """
    Return the sum of gauss functions, defined by the parameters in peaks
    Parameters
        x        : x-values
        peaks    : dictionary {'peak_1':[amp,x0,sigma]}
        bgr_glob : Global background (optional) either constant or array
    Return
        y_calc   : numpy array
    """
    y_calc = bgr_glob
    for peak in peaks:
        y_calc += gauss(x,*peaks[peak])
    return y_calc

def multiPeakFit(x,y,n,peaks=None,bgr_glob=None,verbose=True,x0s=None):
    """
    Fit a peak with n overlapping gaussian functions 
    Parameters:
        x        - x values
        y        - y-values
        n        - number of peaks
        peaks    - (optinal) dictionary of gaussian peak parameters
                   dictionary structure: {'peak_0':[a0,x0,sigma0],
                                          'peak_1':[a0,x0,sigma0],
                                          ...
                                          'peak_n':[a0,x0,sigma0]}
          
                  a0      - amplitude of the n'th gaussian peak
                  x0      - position of the n'th peak in x coordinates
                  sigma0  - standard deviation of the n'th peak
        bgr_glob - (optional) global background constant. Uses y.min() if bgr_glob=None
        verbose  - (optional) bool for toggling print statements (default=True)
    Return 
        peaks    - dictionary of fitted parameters
        bgr      - global background

    Use multiGauss(x,peaks) to get the fitted y-values
    """
    def _multiGauss(x,*p):
        """Parsing function for the optimizer"""
        bgr_glob = p[-1]
        peaks={}
        for i in range(n):
            #          'peak_0':[a0,x0,sigma0]
            peaks[f'peaks_{i}']=p[i*3:(i+1)*3]
        return multiGauss(x,peaks,bgr_glob)
   
    p0 = []
    a0, x0, sigma0 = [],[],[]
    # initialize starting guesses from peaks dictionary, if provided
    if not peaks is None:
        for peak in peaks:
            _a0,_x0,_sigma0 = peaks[peak]
            a0.append(_a0)
            x0.append(_x0)
            sigma0.append(_sigma0)
   
    # find the center of mass index and x-value
    cen_index = np.argmin(np.abs(np.cumsum(y)-np.sum(y)/2))
    cen_x = x[cen_index]
    
    if bgr_glob is None:
        # initial background guess
        bgr_glob = np.nanmin(y)# (y[0]+y[-1])/2
    
    # calculate the numerical integral breadth
    beta = integralBreadth(x,y,bgr_glob)

    # initialize starting values if none were provided
    if len(x0)<1:
        # initial position guess
        # equidistant positions around the center of mass
        if x0s is None:
            x0 = cen_x + (np.linspace(0,beta,n)-beta/2)* float(n>1)
        else:
            x0 = x0s
        
    for i in range(n):    
        if len(a0)<n:
            # initial amplitude guess
            # y-values at the x0 positions
            _a0 = y[np.argmin(np.abs(x-x0[i]))]
            a0.append(_a0)
        if len(sigma0)<n:
            # Guess for sigma based on the numerical integral breadth
            # integral sigma ~ beta/sqrt(2pi)
            _sigma0 = beta/np.sqrt(2*np.pi)/n

            #fwhm0 = beta/n
            #_sigma0 = fwhm0/(2*np.sqrt(2*np.log(2)))
            sigma0.append(_sigma0)
        p0.append(a0[i])
        p0.append(x0[i])
        p0.append(sigma0[i])
    p0.append(bgr_glob)

    # Assume convergence before fit
    convergence = True

    # Fit the peak
    try: #             a0        x0    sigma0        bgr_glob
        lower = [        0.0, x.min(),      0. ]*n + [     0.]
        upper = [y.max()*1.1, x.max(), x.max()-x.min() ]*n + [y.max()]
        bounds = (lower,upper)
        popt,pcov = sci_op.curve_fit(_multiGauss,
                                     x,
                                     y,
                                     p0=p0,
                                     bounds=bounds,
                                     )
        
    except RuntimeError:
        if verbose:
            print('sum(X) fit did not converge!')
        convergence = False
        popt = [np.nan]*len(p0)
        pcov = np.full((len(p0),len(p0)),np.nan)
    except ValueError:
        if verbose:
            print('NaN values')
        convergence = False
        popt = [np.nan]*len(p0)
        pcov = np.full((len(p0),len(p0)),np.nan)

    # restructure fitted parameters is dictionary
    popt = list(popt)
    bgr_glob = popt.pop(-1)
    peaks = {}
    for i in range(n):
        #f'peak_i': [a0,x0,sigma0,bgr]
        peaks[f'peak_{i}']=popt[i*3:(i+1)*3]
    
    # check for peaks switching place
    if verbose:
        keys = sorted(list(peaks.keys()))
        for i in range(len(keys)-1):
            if peaks[keys[i]][1] > peaks[keys[i+1]][1]:
                print(f'{keys[i]} and {keys[i+1]} may have switched place')
    
    return peaks, bgr_glob

def circleGaussFit(chi, y, verbose=False): 
    """
    Parameters
    chi     - chi values in degrees
    y       - y values
    verbose - boolean, verbose
    
    returns -> list of peak parameters.
    """

    use = ~np.isnan(y)
    chi = chi[use]
    y = y[use]
    guess = [0]*4

    guess[3] = np.nanmax(np.nanmin(y),0)
    guess[0] = np.nanmax(y)-guess[3]
    guess[1] = chi[np.nanargmax(y)]
    guess[2] = 0.1
    
    if guess[1] > np.pi:
        guess[1] -= np.pi
        
    lower = [0, 0, 0.01, 0]
    upper = [np.nanmax(y), np.pi,np.pi/5,np.nanmax(y)]

    for i,g in enumerate(guess):
        if g > upper[i]:
            guess[i] = upper[i]
        if g < lower[i]:
            guess[i] = lower[i]

    bounds = (lower, upper)

    if verbose:
        print(f'Guess is: {guess}')
        print(f'Bounds are: {bounds}')
    
    def _circleGaussRes(p, chi, y):
        """
        Computes the residual between the guess paramters p and the test data y
        Parameters:
        chi - chi values in degrees
        y   - intensity in arbritary units
        """
        return (gaussCircle(chi, *p) - y)**2

    if verbose:
        print('Starting fit')

    res_lsq = sci_op.least_squares(
            fun=_circleGaussRes,
            x0=guess,
            bounds=bounds,
            args=(chi,y))
    if verbose:
        print(f'Fit completed:\n\t chi0: {res_lsq.x[1]}\n\t A: {res_lsq.x[0]}\n\t sigma: {res_lsq.x[2]}\n\t bkg: {res_lsq.x[3]}')
    return res_lsq

def parseCircleGaussFit(fit):
    '''
    Parses the result from circleGaussFit.
    Returns an nd array of parameters and a list of their names if fit successful otherwise None.
    Parameters:
    fit - fit from circleGaussFit.
    '''


    names = ['H','DoO','I_tot','I_ori','I_rand', 'FWHM', 'a', 'chi0', 'sigma','bkg']

    if not fit.success:
        return None,names

    result = np.zeros((10,))

    fit_params = np.array(fit.x)
    result[6:10] = fit_params

    result[0] = fit_params[1]/np.pi
    result[3] = 2*np.sqrt(2*np.pi)*np.abs(fit_params[0]*fit_params[2])
    result[4] = fit_params[3] * 2 * np.pi
    result[1] = result[3]/(result[3]+result[4])
    result[2] = (result[3]+result[4])
    result[5] = 4.29193*fit_params[2]

    return result, names




def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fitLine(fun, x, line, mask_line, idx=None, verbose=False):
    """
    Fits every dataset in line using th  function fun with c values c and y_observed y_obs
    returns an array of results from the fit line.

    Parameters:
    fun       - function to use for fitting, should take the parameters x,y_obs
    x         - x values for fit, shape(n,)
    line      - array of observed y values. shape(m,n)
    mask_line - array of 0 and 1. 1 means point will be fitted, 0 means it will not, shape(m,)
    idx       - (optional) the index for parallel computations of multiple lines.
    verbose   - (optional) print statements throughout the process
    
    Returns:
    idx       - (optional) if idx has been set it will be returned.
    results   - array of the results of the fits, shape(n,)
    """

    results = [None]*line.shape[0]
    to_fit = np.sum(mask_line)
    fitted = 0
    if verbose:
        print(f"Fitting {to_fit} points")

    for i, y_obs in enumerate(line):
        if not mask_line[i]:
            continue
        results[i] = fun(x,y_obs)
        fitted += 1 
        if verbose:
            print(f'{timestamp()} -- Done fitting {int(100*fitted/to_fit):03d}%',end='\r')
    if verbose:
        print(f'{timestamp()} -- Completed fitting')

    if idx is None:
        return results
    else:
        return idx, results

