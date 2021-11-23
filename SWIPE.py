# coding: utf-8
"""SWIPE' pitch extraction."""

from pylab import norm, nan, size, fix, polyval, polyfit
from numpy import arange, power, log2, log10, zeros, round_, multiply, divide, reshape, \
newaxis, argwhere, vstack, hanning, transpose, maximum, sqrt, asarray, \
empty, ones, kron, NAN, argmax, concatenate, cos, logical_and, pi, dot, inf

from numpy.matlib import repmat
from scipy.interpolate import interp1d
from misc import transpose1dArray
from matplotlib.pyplot import specgram

def swipep(x, fs, speechFile, plim):
    """Swipe pitch estimation method.

    It estimates the pitch of the vector signal X with sampling frequency Fs
     (in Hertz) every DT seconds. The pitch is estimated by sampling the spectrum
     in the ERB scale using a step of size DERBS ERBs. The pitch is searched
     within the range [PMIN PMAX] (in Hertz) sampled every DLOG2P units in a
    base-2 logarithmic scale of Hertz. The pitch is fine tuned by using parabolic
     interpolation with a resolution of 1/64 of semitone (approx. 1.6 cents).
    Pitches with a strength lower than STHR are treated as undefined.
    """
    
    dt = 0.001
    sTHR = -inf
    dlog2p = 1.0 / 96.0
    dERBs = 0.1

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
        f = asarray(f)
        X1 = transpose(X)

        ip = interp1d(f, X1, kind='linear')(fERBs[:, newaxis]); 
        interpol = reshape(transpose(ip, (2, 0, 1)), (-1, len(ip[0])))
        interpol1 = transpose(interpol)
        M = maximum(0, interpol1)  # Magnitude
        L = sqrt(M)  # Loudness
        
        # Select candidates that use this window size
        if i == (len(ws) - 1):
            j = transpose(argwhere(d - (i + 1) > -1))[0]
            k = transpose(argwhere(d[j] - (i + 1) < 0))[0]
        elif i == 0:
            j = transpose(argwhere(d - (i + 1) < 1))[0]
            k = transpose(argwhere(d[j] - (i + 1) > 0))[0]
        else:
            j = transpose(argwhere(abs(d - (i + 1)) < 1))[0]
            k1 = arange(0, len(j))  # transpose added by KG
            k = transpose(k1)
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j])
        
        # Interpolate at desired times
        if len(Si[0]) > 1:
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
        S[j, :] = S[j, :] + multiply((transpose(kron(ones((len(Si[0]), 1)), mu))), Si)


    # Fine-tune the pitch using parabolic interpolation
    p = empty((len(Si[0]),))
    p[:] = NAN
    s = empty((len(Si[0]),))
    s[:] = NAN
    for j in range(0, len(Si[0])):
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
            ntc = ((tc / tc[1]) - 1) * 2 * pi
            c = polyfit(ntc, (S[I, j]), 2)
            ftc = divide(1, power(2, arange(log2(pc[I[0]]), log2(pc[I[2]]), 0.0013021)))
            nftc = ((ftc / tc[1]) - 1) * 2 * pi
            s[j] = (polyval(c, nftc)).max(0)
            k = argmax(polyval(c, nftc))
            p[j] = 2 ** (log2(pc[I[0]]) + (k - 1) / 768)
    return concatenate((transpose1dArray(t), transpose1dArray(p), transpose1dArray(s)), axis=1)


def pitchStrengthAllCandidates(f, L, pc):
    """Normalize loudness."""
    # warning off MATLAB:divideByZero
    from numpy import sum as sum_
    hh = sum_(multiply(L, L), axis=0)
    ff = transpose(hh[:, newaxis])
    sq = sqrt(ff)

    gh = repmat(sq, len(L), 1)
    gh[gh == 0] = inf
    L = divide(L, gh)
    S = zeros((len(pc), len(L[0])))
    for j in range(0, (len(pc)) - 1):
        S[j, :] = pitchStrengthOneCandidate(f, L, pc[j])
    return S

def is_prime(n):
    """
    Function to check if the number is prime or not.
    """
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def primeArr(n):
    """Return a list containing only prime numbers."""
    return [i for i in range(1, n + 2) if is_prime(i)]


def pitchStrengthOneCandidate(f, L, pc):
    """Normalize the square root of spectrum "L" by applying normalized cosine kernal decaying as 1/sqrt(f)."""
    n = fix(f[-1] / pc - 0.75)
    k = zeros(size(f))
    q = f / pc
    for i in (primeArr(int(n))):
        a = abs(q - i)
        p = a < .25
        k[p] = cos(2 * pi * q[p])
        v = logical_and(.25 < a, a < .75)
        k[v] = cos(2 * pi * q[v]) / 2

    ff = divide(1, f)

    k = (k * sqrt(ff))
    k = k / norm(k[k > 0.0])
    S = dot(transpose(k[:, newaxis]), L)
    return S


def hz2erbs(hz):
    """Converting hz to erbs."""
    erbs = 21.4 * log10(1 + hz / 229)
    return erbs


def erbs2hz(erbs):
    """Converting erbs to hz."""
    hz = (power(10, divide(erbs, 21.4)) - 1) * 229
    return hz