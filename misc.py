# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:06:52 2021

@author: Panagiotis Antivasis
"""
from scipy.io import loadmat

from numpy import asarray, ndarray, zeros, transpose, flip, power, complex128, \
uint8, int8, int16, int32, int64, float16, float32, float64, arange, cos, pi, \
fix, multiply, imag, tile, sqrt, concatenate, fliplr, flipud, ones

from statistics import median
from scipy.signal import ellip, filtfilt

from math import log2

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

def is_prime(n):
    """
    Function to check if the number is prime or not.
    """
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def primes(N):
    '''
    Returns a row vector of prime numbers up to n.
    '''
    return asarray([n for n in range(N) if is_prime(n)])

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
    '''
    An optimized implementation of numpy.hanning.
    '''
    N += 1
    n = arange(1, N)
    return .5*(1 - cos(2*pi*n/N))

def mySpecgram(x,nfft,fs,window,noverlap):
    '''
    A different implementation of specgram
    '''
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

def medfilt(x, p):
    '''
    Performs median filtering of order p.
    
    ----INPUT PARAMETERS----
    1) x: array - The signal 
    2) p: int - The order of the filter
    
    ----OUTPUT PARAMETERS----
    The filtered signal
    '''
    xt = transpose1dArray(x)
    L = len(xt)

    ad = (p-1)/2
    if ad == 0:
        return xt
    from scipy.linalg import toeplitz
    x = concatenate((x[0]*ones(int(ad)), x, x[L-1]*ones(int(ad))))
    
    A = fliplr(toeplitz(flipud(x[0:L]), x[L:L+p-1]))
    
    return [median(a) for a in A]

    