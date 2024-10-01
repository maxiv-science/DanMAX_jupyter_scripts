"""Miscellaneous functions for the DanMAX scripts"""

import numpy as np

hc = 1.23984198e-06 # planck constant * speed of light (eV m)
Cu_ka = 8.046

def keV2A(E):
    """Convert keV to Angstrom"""
    try:
        #hc = 1.23984198e-06 # planck constant * speed of light (eV m)
        lambd = hc/(E*10**3)*10**10
    except ZeroDivisionError:
        lambd = np.full(E.shape,0.0)
    return lambd

def A2keV(lambd):
    """Convert Angstrom to keV"""
    try:
        #hc = 1.23984198e-06 # planck constant * speed of light (eV m)
        E = hc/(lambd*1e-10)*10**-3
    except ZeroDivisionError:
        E = np.full(lambd.shape,0.0)
    return E

def tth2Q(tth,E):
    """Convert 2theta to Q. Provide the energy E in keV"""
    try:
        if len(E)>1:
            E = np.mean(E)
            print(f'More than one energy provided. Using average: {E:.3f} keV')
    except TypeError:
        pass
    try:
        return 4*np.pi*np.sin(tth/2*np.pi/180)/keV2A(E)
    except ZeroDivisionError:
        return np.full(tth.shape,0.0)

def tth2d(tth,E):
    """Convert 2theta to d-spacing. Provide the energy E in keV"""
    try:
        return keV2A(E)/(2*np.sin(tth/2*np.pi/180))
    except ZeroDivisionError:
        return np.full(tth.shape,0.0)    
    
def Q2tth(Q,E):
    """Convert Q to 2theta. Provide the energy E in keV"""
    try:
        if len(E)>1:
            E = np.mean(E)
            print(f'More than one energy provided. Using average: {E:.3f} keV')
    except TypeError:
        pass
    try:
        return 2*np.arcsin(Q*keV2A(E)/(4*np.pi))*180/np.pi
    except ZeroDivisionError:
        return np.full(Q.shape,0.0)

def Q2d(Q,E):
    """Convert d-spacing to Q. Provide the energy E in keV"""
    return tth2d(Q2tth(Q,E),E)

def d2tth(d,E):
    """Convert d-spacing to 2theta. Provide the energy E in keV"""
    try:
        return 2*np.arcsin(keV2A(E)/(2*d))*180/np.pi
    except ZeroDivisionError:
        return np.full(d.shape,0.0)

def d2Q(d,E):
    """Convert d-spacing to Q. Provide the energy E in keV"""
    return tth2Q(d2tth(d,E),E)

def tth2tth(tth0,E1,E0=Cu_ka):
    """Convert tth0 at energy E0 (default=Cu k-alpha) to tth at E1"""
    return Q2tth(tth2Q(tth0,E0),E1)

def reduceArray(a,reduction_factor, axis=0):
    """Reduce the size of an array by step-wise averaging"""
    if a is None:
        return None
    if axis!=0:
        a = np.swapaxes(a,0,axis)
    step = reduction_factor
    last_index = a.shape[0]-step
    a = np.mean([a[i:last_index+i+1:step] for i in range(reduction_factor)],axis=0)
    if axis!=0:
        a = np.swapaxes(a,0,axis)
    return a

def reduceDic(d ,reduction_factor=1, start=None, end=None, axis=0, keys=[],copy_dic=True):
    """Reduce the size of the specified arrays in a dictionary by step-wise averaging"""
    default_keys = ['I', 'cake', 'I_error', 'cake_error','I0', 'time', 'temp', 'energy']
    keys = list(set(default_keys+keys))
    index = np.s_[start:end]
    if reduction_factor==1 and start is None and end is None:
        return d
    if copy_dic:
        d = d.copy()
    for key in keys:
        if key in d.keys() and not d[key] is None:
            if reduction_factor>1:
                d[key] = reduceArray(d[key][index],reduction_factor,axis=axis)
            else:
                d[key] = d[key][index]
    return d

def getVmax(im):
    """
    Return vmax (int) corresponding to the first pixel value with zero counts
    after the maximum value
    """
    im = im[~np.isnan(im)]
    h = np.histogram(im,bins=int(np.max(im)) ,range=(1,int(np.max(im))+1), density=False)
    first_zero = np.argmin(h[0][np.argmax(h[0]):])+np.argmax(h[0])
    return first_zero

def sampleDisplacementCorrection(tth,sdd,d):
    """
    Two theta correction for sample displacement along the beam (flat detector)
    Benjamin S. Hulbert and Waltraud M. Kriven - DOI: 10.1107/S1600576722011360
    Parameters:
        tth  - array - two theta values in degrees
        sdd  - float - sample-to-detector-distance
        d    - flaot - sample displacement, positive towards the detector
    Return:
        corr - array - angular correction in degrees such that 2th_corr = 2th-corr
    """
    corr = np.arctan(d*np.sin(2*tth*np.pi/180)/(2*(sdd-d*np.sin(tth*np.pi/180)**2)))*180/np.pi
    return corr

def edges(x):
    """Return the bin edges for equidistant bin centers"""
    dx = np.mean(np.diff(x))
    return np.append(x,x[-1]+dx)-dx/2

def rebin_1d(x,y,bins=None):
    """
    Re-bin non-equidistant data to equidistant data
        parameters
            x     - Original non-equidistant x-values (m)
            y     - Original non-equidistant y-values (n,m) or (m)
            bins  - (optional) Target equidistant bins
                     if not provided, bins will be generated
                     from x.min() to x.max() with shape (m)
        return
            bins  - Equidistant bins (k)
            y_bin - Binned y-values. Empty bins are filled
                    by linear interpolation. (n,k) or (1,k)
    """
    # unless provided, generate equidistant bin centers from x.min() to x.max()
    if bins is None:
        bins = np.linspace(x.min(),x.max(),x.shape[0])
    # calculate bin edges assuming equidistant bins
    bins_edge = edges(bins)
    y_shape = y.shape
    y = np.atleast_2d(y)
    # re-bin data
    bin_res = binned_statistic(x,y,bins=bins_edge, statistic='mean')
    y_bin = bin_res.statistic
    # fill empty bins by interpolation
    for i,y in enumerate(y_bin):
        y_bin[i] = np.interp(bins,bins[~np.isnan(y)],y[~np.isnan(y)])
    return bins, y_bin


def fft_1d(t, values):
    """Compute the 1D Fourier transform of a signal.

    Parameters
    ----------
    t : array_like
        The times at which the signal is sampled.
    values : array_like
        The values of the signal at each time.

    Returns
    -------
    freqs : ndarray
        The frequencies at which the Fourier transform is computed.
        This is an array of positive frequencies.

    fft_values : ndarray
            The absolute values of the Fourier transform at the corresponding frequencies.
            This is an array of complex numbers.
    """
    # Ensure the input arrays are numpy arrays
    t = np.array(t)
    values = np.array(values)

    # Ensure the input arrays are 1D
    if t.ndim != 1:
        raise ValueError("Input array t must be 1D")
    if values.ndim != 1:
        raise ValueError("Input array values must be 1D")
    
    # Compute the Fourier transform
    fft_values = np.fft.fft(values)
    
    # Compute the corresponding frequencies
    sample_spacing = np.mean(np.diff(t))
    freqs = np.fft.fftfreq(len(values), sample_spacing)

    # Only return the positive frequencies
    fft_values = fft_values[freqs > 0]
    freqs = freqs[freqs > 0]

    # normalize the fft values and take the absolute value
    fft_values = np.abs(fft_values / len(values))

    # Return the frequencies and the Fourier transform values
    return freqs, fft_values