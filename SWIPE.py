# coding: utf-8
"""SWIPE' pitch extraction."""

#import math
#from os import listdir, getcwd
from pylab import norm, nan, size, fix, polyval, polyfit
from matplotlib.pyplot import specgram
#import numpy as np
from numpy import arange, power, log2, log10, zeros, round_, multiply, divide, reshape, \
newaxis, argwhere, vstack, hanning, array, transpose, maximum, sqrt, asarray, \
empty, ones, kron, NAN, argmax, concatenate, cos, logical_and, pi, dot #, seterr, expand_dims

from numpy.matlib import repmat
#from numpy import matlib
#from scipy.io import wavfile
#from scipy import signal
#from scipy import interpolate
from scipy.interpolate import interp1d
from misc import transpose1dArray

WAVE_OUTPUT_FILENAME = 'Audio.wav'  # Provide here the path to an audio


def swipep(x, fs, plim, dt, sTHR):
    """Swipe pitch estimation method.

    It estimates the pitch of the vector signal X with sampling frequency Fs
     (in Hertz) every DT seconds. The pitch is estimated by sampling the spectrum
     in the ERB scale using a step of size DERBS ERBs. The pitch is searched
     within the range [PMIN PMAX] (in Hertz) sampled every DLOG2P units in a
    base-2 logarithmic scale of Hertz. The pitch is fine tuned by using parabolic
     interpolation with a resolution of 1/64 of semitone (approx. 1.6 cents).
    Pitches with a strength lower than STHR are treated as undefined.
    """
    if not plim:
        plim = [30, 5000]
    if not dt:
        dt = 0.01
    dlog2p = 1.0 / 96.0
    dERBs = 0.1
    if not sTHR:
        sTHR = -float('Inf')

    t = arange(0, len(x) / float(fs), dt)    # Times
    dc = 4   # Hop size (in cycles)
    K = 2   # Parameter k for Hann window
    # Define pitch candidates
    log2pc = arange(log2(plim[0]), log2(plim[len(plim) - 1]), dlog2p)
    pc = power(2, log2pc)

    S = zeros(shape=(len(pc), len(t)))   # Pitch strength matrix

    # Determine P2-WSs
    logWs = round_(log2(multiply(4 * K, (divide(float(fs), plim)))))
    ws = power(2, arange(logWs[1 - 1], logWs[2 - 1] - 1, -1))  # P2-WSs
    pO = 4 * K * divide(fs, ws)   # Optimal pitches for P2-WSs
    # Determine window sizes used by each pitch candidate
    d = 1 + log2pc - log2(multiply(4 * K, (divide(fs, ws[1 - 1]))))
    # Create ERBs spaced frequencies (in Hertz)
    fERBs = erbs2hz(arange(hz2erbs(pc[1 - 1] / 4), hz2erbs(fs / 2), dERBs))

    for i in range(0, len(ws)):
        ws[i] = int(ws[i])
        # for i in range(0, 1):
        dn = round(dc * fs / pO[i])  # Hop size (in samples)
        # Zero pad signal
        will = zeros((int(ws[i] / 2), 1))
        learn = reshape(x, -1, order='F')[:, newaxis]
        mir = zeros((int(dn + ws[i] / 2), 1))
        xzp = vstack((will, learn, mir))
        xk = reshape(xzp, len(xzp), order='F')
        # Compute spectrum
        w = hanning(ws[i])  # Hann window
        o = max(0, round(ws[i] - dn))  # Window overlap
        [X, f, ti, im] = specgram(xk, NFFT=int(ws[i]), Fs=fs, window=w, noverlap=int(o))

        # Interpolate at equidistant ERBs steps
        f = array(f)
        X1 = transpose(X)

        ip = interp1d(f, X1, kind='linear')(fERBs[:, newaxis])
        interpol = ip.transpose(2, 0, 1).reshape(-1, ip.shape[1])
        interpol1 = transpose(interpol)
        M = maximum(0, interpol1)  # Magnitude
        L = sqrt(M)  # Loudness
        # Select candidates that use this window size
        if i == (len(ws) - 1):
            j = argwhere(d - (i + 1) > -1).transpose()[0]
            k = argwhere(d[j] - (i + 1) < 0).transpose()[0]
        elif i == 0:
            j = argwhere(d - (i + 1) < 1).transpose()[0]
            k = argwhere(d[j] - (i + 1) > 0).transpose()[0]
        else:
            j = argwhere(abs(d - (i + 1)) < 1).transpose()[0]
            k1 = arange(0, len(j))  # transpose added by KG
            k = transpose(k1)
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j])
        # Interpolate at desired times
        if Si.shape[1] > 1:
            tf = []
            tf = ti.tolist()
            tf.insert(0, 0)
            del tf[-1]
            ti = asarray(tf)
            Si = interp1d(ti, Si, 'linear', fill_value=nan)(t)
        else:
            Si = repmat(float('NaN'), len(Si), len(t))
        lambda1 = d[j[k]] - (i + 1)
        mu = ones(size(j))
        mu[k] = 1 - abs(lambda1)
        S[j, :] = S[j, :] + multiply(((kron(ones((Si.shape[1], 1)), mu)).transpose()), Si)

    # Fine-tune the pitch using parabolic interpolation
    p = empty((Si.shape[1],))
    p[:] = NAN
    s = empty((Si.shape[1],))
    s[:] = NAN
    for j in range(0, Si.shape[1]):
        s[j] = (S[:, j]).max(0)
        i = argmax(S[:, j])
        if s[j] < sTHR:
            continue
        if i == 0:
            p[j] = pc[0]
        elif i == len(pc) - 1:
            p[j] = pc[0]
        else:
            I = arange(i - 1, i + 2)
            tc = divide(1, pc[I])
            # print "pc[I]", pc[I]
            # print "tc", tc
            ntc = ((tc / tc[1]) - 1) * 2 * pi
            # print "S[I,j]: ", shape(S[I,j])
            # with warnings.catch_warnings():
            # warnings.filterwarnings('error')
            # try:
            c = polyfit(ntc, (S[I, j]), 2)
            # print "c: ", c
            ftc = divide(1, power(2, arange(log2(pc[I[0]]), log2(pc[I[2]]), 0.0013021)))
            nftc = ((ftc / tc[1]) - 1) * 2 * pi
            s[j] = (polyval(c, nftc)).max(0)
            k = argmax(polyval(c, nftc))
            # except RankWarning:
            # print ("not enough data")
            p[j] = 2 ** (log2(pc[I[0]]) + (k - 1) / 768)
    #p[isnan(s) - 1] = float('NaN')  # added by KG for 0s
    return concatenate((transpose1dArray(t), transpose1dArray(p), transpose1dArray(s)), axis=1)


def pitchStrengthAllCandidates(f, L, pc):
    """Normalize loudness."""
    # warning off MATLAB:divideByZero
    import numpy.sum
    hh = numpy.sum(multiply(L, L), axis=0)
    ff = (hh[:, newaxis]).transpose()
    sq = sqrt(ff)

    gh = repmat(sq, len(L), 1)
    L = divide(L, gh)
    S = zeros((len(pc), len(L[0])))
    for j in range(0, (len(pc)) - 1):
        S[j, :] = pitchStrengthOneCandidate(f, L, pc[j])
    return S

numArr = []


def is_prime(n):
    """Function to check if the number is prime or not."""
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def primeArr(n):
    """Return a list containing only prime numbers."""
    for num in range(1, n + 2):
        if is_prime(num):
            numArr.append(num)
    #jg = (expand_dims(numArr, axis=1)).transpose()
    return numArr


def pitchStrengthOneCandidate(f, L, pc):
    """Normalize the square root of spectrum "L" by applying normalized cosine kernal decaying as 1/sqrt(f)."""
    n = fix(f[-1] / pc - 0.75)
    k = zeros(size(f))
    q = f / pc
    for i in (primeArr(int(n))):
        # print "i is:",i
        a = abs(q - i)
        p = a < .25
        k[argwhere(p)] = cos(2 * pi * q[argwhere(p)])
        v = logical_and(.25 < a, a < .75)
        #pl = cos(2 * pi * q[argwhere(v)]) / 2
        k[argwhere(v)] = cos(2 * pi * q[argwhere(v)]) / 2

    ff = divide(1, f)

    k = (k * sqrt(ff))
    k = k / norm(k[k > 0.0])
    S = dot((k[:, newaxis]).transpose(), L)
    return S


def hz2erbs(hz):
    """Converting hz to erbs."""
    erbs = 21.4 * log10(1 + hz / 229)
    return erbs


def erbs2hz(erbs):
    """Converting erbs to hz."""
    hz = (power(10, divide(erbs, 21.4)) - 1) * 229
    return hz


'''def swipe(audioPath):
    """Read the audio file and output the pitches and pitch contour."""
    print("Swipe running", audioPath)
    fs, x = wavfile.read(audioPath)
    seterr(divide='ignore', invalid='ignore')
    p, t, s = swipep(x, fs, [100, 600], 0.001, 0.3)
    print("Pitches: ", p)
    fig = plt.figure()
    plt.plot(p)
    fig.savefig('hummed.png')
    plt.show()  # show in a window of contour on UI

#swipe(WAVE_OUTPUT_FILENAME)'''
