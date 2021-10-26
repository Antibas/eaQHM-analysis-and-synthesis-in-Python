# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:06:52 2021

@author: Panagiotis Antivasis
"""
from scipy.io import loadmat

from numpy import asarray, ndarray, zeros, transpose, flip, power, complex128, \
uint8, int8, int16, int32, int64, float16, float32, float64, arange, cos, pi, \
fix, multiply, imag, tile

from scipy.signal import ellip, filtfilt

from math import log2

from sympy import isprime

from copy import deepcopy

normalize = 32768

def matToObject(filename: str):
    '''
    Loads a parameter file with name filename and
    converts it into a dictionary.
    
    ----INPUT PARAMETERS----
    filename: the location of a parameter file.
    
    ----OUTPUT PARAMETERS----
    A dictionary containing all stored data in 
    that parameter file.
    '''
    
    param = loadmat(filename)
    del param["__header__"]
    del param["__version__"]
    del param["__globals__"]
    
    obj = {}
    for key, value in param.items():
        obj[key] = singlelize(value)
    
    return obj


def loadOptions(filename: str = "", ignoreSWIPEP: bool = False):
    '''
    Loads a parameter file with name filename to
    load the options of eaQHManalysis. 
    The file must contain a data named 'opt'
    with 10 integers, each one mapped to a respective
    option parameter (see 'help eaQHManalysis').
    
    If no string or an empty string is given, the
    dictionary is loaded with the default values.
    
    ----INPUT PARAMETERS----
    filename: the location of a parameter file.
    
    ----OUTPUT PARAMETERS----
    A dictionary containing all option parameters 
    for eaQHManalysis.
    '''
    
    if filename != "":
        opt = loadmat(filename)['opt']
        
        obj = {
                "fullWaveform": opt[0][0][0][0][0] == 1,  
                "fullBand": opt[0][0][2][0][0] == 1,
                "extended_aQHM": opt[0][0][3][0][0] == 1,
                "highPassFilter": opt[0][0][6][0][0] == 1,
                "SWIPEP": opt[0][0][8][0][0] == 1 and not ignoreSWIPEP,
                "numPartials": opt[0][0][10][0][0]
            }
    else:
        obj = {
                "fullWaveform": True,  
                "fullBand": True,
                "extended_aQHM": True,
                "highPassFilter": False,
                "SWIPEP": True and not ignoreSWIPEP,
                "numPartials": 0
            }
    return obj

def loadParameters(filename: str):
    '''
    Loads a parameter file with name filename to
    load all other parameters of eaQHManalysis. 
    Default values are also assigned to some parameters
    if not loaded.
    
    ----INPUT PARAMETERS----
    filename: the location of a parameter file.
    
    ----OUTPUT PARAMETERS----
    A dictionary containing all parameters 
    for eaQHManalysis.
    '''
    
    param = matToObject(filename)
    if not "step" in param:
        param["step"] = 15
    if not "adpt" in param:
        param["adpt"] = 6
    if not "NoP" in param:
        param["NoP"] = 3
    if not "PAW" in param:
        param["PAW"] = 32
    
    
    return param

def transpose1dArray(x):
    '''
    Transposes a 1d array-like.
    
    WARNING!!!: Does not work with 2d containers.
    
    ----INPUT PARAMETERS----
    x: A 1d array-like.
    
    ----OUTPUT PARAMETERS----
    The transposition of that array-like as numpy.ndarray
    '''
    
    return asarray([x]).T

def mytranspose(x):
    '''
    Either uses numpy.transpose or transpose1dArray
    to array or list x, depending on if it's 1d or not.
    
    ----INPUT PARAMETERS----
    x: An array or list.
    
    ----OUTPUT PARAMETERS----
    The transposition of that array or list.
    '''
    
    if len(x) == 1:
        return transpose1dArray(x)
    return singlelize(transpose(x))

def end(a):
    '''
    Returns the last element of numpy.ndarray a.
    
    E.g. for [1, 2, 3, 4, 5] it will return 5, while
    for [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]] it will return 10.
    
    ----INPUT PARAMETERS----
    a: A numpy.ndarray array.
    
    ----OUTPUT PARAMETERS----
    The last element of array a.
    '''
    
    if isinstance(a, ndarray):
        return end(a[len(a)-1])
    return a
   
def arrayByIndex(idxs: ndarray, value):
    '''
    Similarly with MATLAB, maps all values of array-like or int value 
    to the respective indices in numpy.ndarray idx in 
    a new numpy.ndarray object. Non-mapped values are replaced with 0.
    
    E.g arrayByIndex(asarray([0,2,4,6]), [1,2,3,4]) returns 
    array([1, 0, 2, 0, 3, 0, 4])
    
    
    ----INPUT PARAMETERS----
    1) idxs: A numpy.ndarray containing the indices to be mapped.
    2) value: An array-like or int containing the values to be mapped.
    
    ----OUTPUT PARAMETERS----
    A numpy.ndarray containing all mapped values.
    '''
    
    if isinstance(value, int):
        r = zeros(int(end(idxs))+1, dtype=type(value[0]))
        for idx in idxs:
            r[int(idx)] = value
        return r
    
    value = flip(value)
    r = zeros(int(end(idxs))+1, dtype=type(value[0]))
    for idx in idxs:
        r[int(idx)], value = value[-1], value[:-1]
    return r

def singlelize(a): 
    '''
    Simplifies array-like a that contains only one element.
    E.g singlelize([1]) returns 1 while singlelize([[1,2,3]]) 
    return [1,2,3].
    
    ----INPUT PARAMETERS----
    a: An array-like object.
    
    ----OUTPUT PARAMETERS----
    A simplified array-like object.
    '''
    
    if not (isinstance(a, ndarray) or isinstance(a, list)):
        return a
    
    if len(a) == 1:
        return singlelize(a[0])
    
    return a

def apply(v, lamda):
    '''
    Applies a function lamda on a certain iterable of numbers.
    
    ----INPUT PARAMETERS----
    1) v: iterable - The iterable.
    2) lamda: function - The function to be applied.
    
    ----OUTPUT PARAMETERS----
    The iterable after the lamda function is applied in each element.
    '''
    if isNum(v):
        return lamda(v)
    if isContainer(v):
        vv = []
        for i in range(len(v)):
            vv.append(apply(v[i], lamda))
        return asarray(vv)
    raise TypeError(type(v))   

def primes(N):
    '''
    Returns a row vector of prime numbers up to n.
    '''
    p = []
    for n in range(N):
        if isprime(n):
            p.append(n)
    return asarray(p)

def hz2erbs(hz):
    '''
    Converts Hz to erbs.
    '''
    return 6.44 * ( log2( 229 + hz ) - 7.84 )

def erbs2hz(erbs):
    '''
    Converts erbs to Hz.
    '''
    return power(2, erbs/6.44 + 7.84) - 229

def isComplex(a):
    '''
    Returns if a is a complex number.
    '''
    return isinstance(a, complex128) or isinstance(a, complex)

def isNum(a):
    '''
    Returns if a is a number.
    '''
    return isinstance(a, float) or isinstance(a, int) \
        or isinstance(a, float16) or isinstance(a, int16) or isinstance(a, int8) \
        or isinstance(a, uint8) \
        or isinstance(a, float32) or isinstance(a, int32) \
        or isinstance(a, float64) or isinstance(a, int64) \
        or isComplex(a)

def isContainer(a):
    '''
    Returns if a is a container.
    '''
    return isinstance(a, ndarray) or isinstance(a, list)

def isEmpty(a):
    '''
    Returns if a is empty. If a is not an iterable, 0 is returned
    '''
    if isContainer(a):
        return len(a) == 0
    return False

def ellipFilter(s, fs, fc, ftype='highpass'):
    '''
    Performs a basic filter on a signal
    
    ----INPUT PARAMETERS----
    1) s: array - The signal.
    2) fs: int - The sampling frequency.
    3) fc: int - The cutoff frequency.
    4) ftype: str (optional) - The type of the filter. 
    Default value is 'highpass'.
    
    ----OUTPUT PARAMETERS----
    The filtered signal
    '''
    bHigh, aHigh = ellip(6, .5, 60, 2*fc/fs, ftype)
    return filtfilt(bHigh, aHigh, s)

def arrayMax(n, a):
    a2 = deepcopy(a)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] < n:
                a2[i][j] = n
    return a2

def myHann(N):
    N += 1
    n = arange(1, N)
    return .5*(1 - cos(2*pi*n/N))

def spline_interp(x, y, xn):
    N = len(x)
    M = len(xn)
    
    y2 = spline(x, y)
    yf = zeros(M)
    
    for i in range(M):
        tmp = xn[i]
        if tmp < x[1]:
            yf[i] = y[1]
            continue
        elif tmp > x[N-1]:
            yf[i] = y[N-1]
            continue
        else:
            yf[i] = splint(x, y, y2, tmp)
    
    return yf

def spline(x, y):
    n = len(x)
    u = zeros(n-1)
    
    y2 = zeros(n)
    
    y2[1] = 0.5
    u[1] = (3/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1]))
    
    for i in range(2, n-1):
        sig = (x[i] - x[i-1])/(x[i+1] - x[i-1])
        p = sig*y2[i-1] + 2.0
        y2[i] = (sig - 1.0)/p
        u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1])
        u[i] = (6.0*u[i]/(x[i+1] - x[i-1]) - sig*u[i-1])/p
    
    qn = 0.5
    un = (3.0/(x[n-1] - x[n-2])) * (-(y[n-1] - y[n-2])/(x[n-1] - x[n-2]))
    
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0)
    
    for k in range(n-1, 1, -1):
        y2[k-1] = y2[k-1]*y2[k] + u[k-1]
    
    return y2

def splint(x, y, y2, xv):
    n = len(x)
    klo = 1 
    khi = n
    
    while khi - klo > 1:
        k = (khi + klo) >> 1
        if (x[k] > xv): 
            khi = k
        else:
            klo = k
    
    h = x[khi] - x[klo]
    
    if h == 0:
        raise ValueError("Bad x input to routine splint()!\n")
    
    a = (x[khi] - xv)/h
    b = (xv - x[klo])/h
    
    y = a*y[klo] + b*y[khi] + ((a*a*a-a)*y2[klo] + (b*b*b-b)*y2[khi])*(h*h)/6.0
    return y

def mySpecgram(x,nfft,fs,window,noverlap):
    from numpy.fft import fft
                
    nx = len(x)
    
    nwind = len(window)
    
    if nx < nwind:
        x[nwind-1] = 0
        nx = nwind
        
    ncol = int(fix((nx-noverlap)/(nwind-noverlap)))
    
    colindex = arange(0, ncol)*(nwind-noverlap)
    
    rowindex = arange(nwind)
    
    if len(x)<(nwind+colindex[ncol-1]-1):
        x[nwind+colindex[ncol-1]-2] = 0
    
    #y = zeros((nwind, ncol))
    y = x[tile(transpose1dArray(rowindex), ncol)+tile(transpose1dArray(colindex), nwind).transpose()]
    
    y2 = multiply(tile(transpose1dArray(window), ncol), y)
    
    y3 = fft(y2.transpose(),nfft).transpose()
    
    if not any(imag(x)):
        if nfft % 2 == 0:
            select = arange(0, (nfft+1)/2, dtype=int)
        else:
            select = arange(0, (nfft)/2, dtype=int)
    
        y3 = y3[select, :]
    else:
        select = arange(0, nfft)
    
    f = (select)*fs/nfft
    t = colindex/fs
    return y3, f, t
    