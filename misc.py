# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:06:52 2021

@author: Panagiotis Antivasis
"""
from scipy.io import loadmat

from numpy import asarray, ndarray, zeros, transpose, flip, power, complex128, \
uint8, int8, int16, int32, int64, float16, float32, float64, arange, cos, pi

from scipy.signal import ellip, filtfilt

from math import log2

from sympy import isprime

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


def loadOptions(filename: str = ""):
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
                "SWIPEP": opt[0][0][8][0][0] == 1,
                "numPartials": opt[0][0][10][0][0]
            }
    else:
        obj = {
                "fullWaveform": True,  
                "fullBand": True,
                "extended_aQHM": True,
                "highPassFilter": False,
                "SWIPEP": True,
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

def myHann(N):
    n = arange(0, N+1)
    return .5*(1 - cos(2*pi*n/N))
    