# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:00:19 2021

@author: Panagiotis Antivasis
"""
from time import time, strftime, gmtime

from structs import Deterministic, Frame
from numpy import arange, zeros, blackman, hamming, \
argwhere, insert, flipud, asarray, append, multiply, \
real, imag, pi, divide, log10, angle, diff, unwrap, sin, cos, \
std, concatenate, tile, dot, ndarray, transpose, conjugate, ones

from numpy.linalg import inv

from scipy.interpolate import interp1d
from scipy.signal import lfilter
from scipy.io.wavfile import read

from misc import arrayByIndex, mytranspose, end, transpose1dArray, normalize, \
isContainer, isEmpty, ellipFilter, \
medfilt, min_interp_size

from copy import deepcopy

from tqdm import tqdm

from warnings import filterwarnings

from SWIPE import swipep



def eaQHMAnalysisAndSynthesis(speechFile: str, gender: str or tuple = 'other', step: int = 15,
                  maxAdpt: int = 10, pitchPeriods: int = 3, analysisWindow: int = 32, fullWaveform: bool = True,
                  fullBand: bool = True, eaQHM: bool = True, fc: int = 0, partials: int = 0,
                  extraInfo: bool = False, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs adaptive Quasi-Harmonic Analysis of Speech
    using the extended adaptive Quasi-Harmonic Model and decomposes 
    speech into AM-FM components according to that model,
    while iteratively refining it.

    Parameters
    ----------
    speechFile : str
        The location of the mono .wav file to be analysed.
    gender : str or tuple, optional
        The gender of the speaker, defining the pitch limit of SWIPEP.
        If str, 'male' limit is [70, 180], 'female' is [70, 180] and 'child' can also be used for [300, 600]. Any other string is [70, 500]. 
        If tuple, gender[0] is the minimum and gender[1] the maximum pitch limit.
        The default is 'other'.
    step : int, optional
        The step size of the processing in samples. The default is 15.
    maxAdpt : int, optional
        The maximum number of adaptations allowed. The default is 10.
    pitchPeriods : int, optional
        The number of analysis window size, in pitch periods. The default is 3.
    analysisWindow : int, optional
        The samples of the pitch analysis window, where the analysis starts. The default is 32.
    fullWaveform : bool, optional
        Determines if a full waveform length analysis will be performed. The default is True.
    fullBand : bool, optional
        Determines if a full band analysis-in-voiced-frames will be performed. The default is True.
    eaQHM : bool, optional
        Determines if an adaptive Quasi-Harmonic Model or an extended adaptive Quasi-Harmonic Model will be used. The default is True.
    fc : int, optional
        Applies a high pass filtering at the specified Hz before the analysis starts. If <= 0, no filter is applied. The default is 0.
    partials : int, optional
        The number of partials to be used. If <= 0, it is determined by the pitch estimations. The default is 0.
    extraInfo : bool, optional
        Determines if 
    printPrompts : bool, optional
        Determines if prompts of this process will be printed. The default is True.
    loadingScreen : bool, optional
        Determines if a tqdm loading screen will be displayed in the console. The default is True.

    Returns
    -------
    s_recon : (length) array_like
        The refined signal.
    SRER : list
        An array containing all the adaptation numbers of the signal.
    DetComponents : (No_ti) array_like, optional
        The Deterministic components of the signal. An array containing elements of Deterministic-class type. 
        Only returned if extraInfo == True.
    Kmax : int, optional
        The number of partials used. Only returned if extraInfo == True.
    Fmax : int, optional
        The maximum frequency of the analysis. Only returned if extraInfo == True.
    endTime : time, optional
        The total time the process takes. Only returned if extraInfo == True.
    '''
    filterwarnings("ignore")
    startTime = time()
    
    fs, s = read(speechFile)
    s = transpose1dArray(s/normalize)
    length = len(s)
    
    if fc > 0:
        s = transpose(ellipFilter(transpose(s), fs, fc))
        
    s2 = deepcopy(s)
        
    if isinstance(gender, tuple):
        f0min = gender[0]
        f0max = gender[1]
    elif gender == 'male':
        f0min = 70
        f0max = 180
    elif gender == 'female':
        f0min = 160
        f0max = 300
    elif gender == 'child':
        f0min = 300
        f0max = 600
    else:
        f0min = 70
        f0max = 500
    
    f0s = swipep(transpose(s2)[0], fs, speechFile, [f0min, f0max])

    f0s = getLinear(f0s, arange(0, len(s2)-1, round(fs*5/1000))/fs)
       
    if fullBand:
        Fmax = int(fs/2-200)
        
        if partials > 0:
            Kmax = partials
        else:
            Kmax = int(round(Fmax/min(f0s[:,1])) + 10)
    else:
        Fmax = int(fs/2-2000)
        Kmax = int(round(Fmax/min(f0s[:,1])) + 10) 
    
    
    analysisWindowSamples = analysisWindow*step
    
    frames, frame_step = voicedUnvoicedFrames(s, fs, gender)

    if not fullWaveform:
        sp_i = []
        ss = zeros((len(s), 1))
        for f in frames:
            if f.isSpeech and f.isVoiced:
                sp_i.append(f.ti)
            else:
                if not isEmpty(sp_i):
                    sp_v = arange(sp_i[0]-frame_step, end(sp_i)+frame_step+1)
                    ss[sp_v] = s[sp_v]
                    sp_i.clear()
        deterministic_part = ss 
    else:
        for i, f in enumerate(frames):
            if f.ti > analysisWindowSamples/2 and f.isSpeech and (not f.isVoiced) and f.ti < length - analysisWindowSamples/2:
                frames[i].isVoiced = True
            if f.ti > analysisWindowSamples/2 and (not f.isSpeech) and (not f.isVoiced) and f.ti < length - analysisWindowSamples/2:
                frames[i].isSpeech = True
                frames[i].isVoiced = True
        deterministic_part = s

    ti = arange(1,length,step)
    No_ti = len(ti)

    DetComponents = [Deterministic() for _ in range(No_ti)]
    SRER = []
    
    
    framei = (ti/frame_step)
    framei_int = framei.astype(int)
    window_lengths = zeros(No_ti, int)

    fm_current = zeros((length, Kmax), float)
    am_current = zeros((length, Kmax), float)
    std_det = std(deterministic_part)
    
    for a in range(maxAdpt+1):
        adptStartTime = time()
        if printPrompts:
            print('---- Adaptation No. {} ----\n'.format(a))
            
        a0_recon = zeros(length, float)
        am_recon = zeros((length, Kmax), float)
        fm_recon = zeros((length, Kmax), float)
        ph_recon = zeros((length, Kmax), float)
        
        if loadingScreen:
            analysisloop = tqdm(total=No_ti, position=0, leave=True)
            
        for i, tith in enumerate(ti):
            if loadingScreen:
                analysisloop.set_description("Analysis".format(i))
            
            if tith > analysisWindowSamples and tith < length-analysisWindowSamples:
                if frames[framei_int[i]-1].isVoiced and frames[framei_int[i]].isVoiced: 
                    if a == 0:
                        framei_dec = framei[i] - framei_int[i]
                        
                        f0 = (1-framei_dec)*f0s[framei_int[i]-1][1] + framei_dec*f0s[framei_int[i]][1]
                        
                        K = int(min(Kmax, int(Fmax/f0)))
                        
                        f0range = arange(-K,K+1)*f0
                        
                        window_lengths[i] = max(120, round((pitchPeriods/2)*(fs/f0))) 
                        
                        window_range = arange(-window_lengths[i]-1,window_lengths[i]) 
                        window = blackman(2*window_lengths[i]+1)
                        
                        amplitudes, slopes, fmismatch = iqhmLS_complexamps(s[window_range + tith], f0range, window, fs)
                    else:
                        window_range = arange(-window_lengths[i]-1,window_lengths[i])
                        window = hamming(2*window_lengths[i]+1)
                            
                        fm_current_nonzeros = argwhere(fm_current[tith-1])
                        
                        if len(fm_current_nonzeros) == 0:
                            fm_current_nonzeros = 0
                            
                            K = 1
                            
                            fm_current[tith-1, fm_current_nonzeros] = 140
                            am_current[tith-1, fm_current_nonzeros] = 10e-4
                            
                            fm = fm_current[tith + window_range, fm_current_nonzeros]
                            am = am_current[tith + window_range, fm_current_nonzeros]
                            
                            fm_len = len(fm)
                                
                            fm_zeros = argwhere(fm == 0)
                            fm_nonzeros = argwhere(fm)
                            
                            if len(fm_zeros) != 0:
                                fm_zeros_index = fm_zeros[0][0]
                                fm_nonzeros_index = fm_nonzeros[0][0]
                                if fm_zeros_index == 0:
                                    fm[fm_zeros_index]= fm[fm_nonzeros_index]
                                    am[fm_zeros_index] = am[fm_nonzeros_index]
                                   
                                    fm_nonzeros = insert(fm_nonzeros, 0, fm_zeros_index)
                                
                                fm_zeros_index = end(fm_zeros)
                                fm_nonzeros_index = end(fm_nonzeros)
                                if fm_zeros_index == fm_len-1:
                                    fm[fm_zeros_index] = fm[fm_nonzeros_index]
                                    am[fm_zeros_index] = am[fm_nonzeros_index]
                                   
                                    fm_nonzeros = append(fm_nonzeros, fm_zeros_index)
                            
                            x_new = arange(0, fm_len)
                            if len(transpose(fm_nonzeros)) == 1:
                                fm_nonzeros = transpose(fm_nonzeros)[0]
                                
                            fm = interp1d(fm_nonzeros, fm[fm_nonzeros])(x_new)
                            am = interp1d(fm_nonzeros, am[fm_nonzeros])(x_new)
                        else:
                            fm = transpose(fm_current[tith + window_range, fm_current_nonzeros])
                            am = transpose(am_current[tith + window_range, fm_current_nonzeros])
                            Kend = len(fm_current_nonzeros)
                        
                            K = end(fm_current_nonzeros)+1
                        
                            for k in range(Kend):
                                fm_len = len(fm[:, k])
                                
                                fm_zeros = argwhere(fm[:, k] == 0)
                                fm_nonzeros = argwhere(fm[:, k])
                                
                                if len(fm_zeros) != 0:
                                    fm_zeros_index = fm_zeros[0][0]
                                    fm_nonzeros_index = fm_nonzeros[0][0]
                                    if fm_zeros_index == 0:
                                        fm[fm_zeros_index][k]= fm[fm_nonzeros_index][k]
                                        am[fm_zeros_index][k] = am[fm_nonzeros_index][k]
                                       
                                        fm_nonzeros = insert(fm_nonzeros, 0, fm_zeros_index)
                                    
                                    fm_zeros_index = end(fm_zeros)
                                    fm_nonzeros_index = end(fm_nonzeros)
                                    if fm_zeros_index == fm_len-1:
                                        fm[fm_zeros_index][k] = fm[fm_nonzeros_index][k]
                                        am[fm_zeros_index][k] = am[fm_nonzeros_index][k]
                                       
                                        fm_nonzeros = append(fm_nonzeros, fm_zeros_index)
                                
                                x_new = arange(0, fm_len)
                                if len(transpose(fm_nonzeros)) == 1:
                                    fm_nonzeros = transpose(fm_nonzeros)[0]
                                    
                                fm[:, k] = interp1d(fm_nonzeros, fm[fm_nonzeros, k])(x_new)
                                am[:, k] = interp1d(fm_nonzeros, am[fm_nonzeros, k])(x_new)
                        
                        if isinstance(fm_current_nonzeros, ndarray):
                            fm_current_nonzeros = concatenate((transpose(-flipud(fm_current_nonzeros+1))[0], [0], transpose(fm_current_nonzeros)[0]+1)) + K
                            tmp_zeros = zeros((fm_len, 1), float)
                        
                            fm = concatenate((-flipud(fm), tmp_zeros, fm), axis=1)
                            am = concatenate((flipud(am), tmp_zeros, am), axis=1)
                        else:
                            fm_current_nonzeros = asarray([K-1, K, K+1])
                        
                            tmp_zeros = zeros((fm_len, 1), float)
                        
                            fm = concatenate((-flipud(transpose1dArray(fm)), tmp_zeros, transpose1dArray(fm)), axis=1)
                            am = concatenate((flipud(transpose1dArray(am)), tmp_zeros, transpose1dArray(am)), axis=1)
                        
                        if not eaQHM:
                            amplitudes_tmp, slopes_tmp = aqhmLS_complexamps(s[window_range + tith], fm, window, fs)
                        else:
                            amplitudes_tmp, slopes_tmp = eaqhmLS_complexamps(s[window_range + tith], am, fm, window, fs)
                        
                        fmismatch_tmp = fs/(2*pi)*divide(multiply(real(amplitudes_tmp), imag(slopes_tmp)) - multiply(imag(amplitudes_tmp), real(slopes_tmp)), abs(amplitudes_tmp)**2)
                        
                        amplitudes = arrayByIndex(fm_current_nonzeros, transpose(amplitudes_tmp)[0])
                        slopes = arrayByIndex(fm_current_nonzeros, transpose(slopes_tmp)[0])
                        fmismatch = arrayByIndex(fm_current_nonzeros, transpose(fmismatch_tmp)[0])
                        
                    a0_recon[tith-1] = real(amplitudes[K])
                    
                    amplitudes = mytranspose(amplitudes[K+1:2*K+1])
                    slopes = mytranspose(slopes[K+1:2*K+1])
                    fmismatch = mytranspose(fmismatch[K+1:2*K+1])
                    
                    amplitude_log_max = 20*log10(max(abs(amplitudes)))-150
                    h = f0/(a+1)
                    
                    for k in range(K):
                        amplitude_log = 20*log10(abs(amplitudes[k]))
                        
                        if amplitude_log > amplitude_log_max and abs(fmismatch[k]) < h:
                            am_recon[tith-1][k] = abs(amplitudes[k])
                            ph_recon[tith-1][k] = angle(amplitudes[k])
                                
                            if a == 0:
                                fm_recon[tith-1][k] = (k+1)*f0 + fmismatch[k]
                            elif f0 > f0min:
                                fm_recon[tith-1][k] = fm_current[tith-1][k] + fmismatch[k]
                            else:
                                fm_recon[tith-1][k] = fm_current[tith-1][k]
                    DetComponents[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=True)
                else:
                   DetComponents[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=False)
            else:
                DetComponents[i] = Deterministic(ti=tith-1, isSpeech=False, isVoiced=False)
            
            if loadingScreen:
                analysisloop.update(1)
        
        if loadingScreen:
            analysisloop.close()
        
        fm_current = zeros((length, Kmax), float)
        am_current = zeros((length, Kmax), float)
        
        a0_recon = interp1d(ti-1, a0_recon[ti-1], kind=3, fill_value="extrapolate")(arange(0, length))

        
        if loadingScreen:
            interploop = tqdm(total=Kmax, position=0, leave=True)
            
        for k in range(Kmax): 
            if loadingScreen:
                interploop.set_description("Interpolation".format(k))
            
            am_recon_nonzeros = argwhere(am_recon[:, k]) 
            
            diff_am_recon_nonzeros = diff(concatenate(([0], transpose(am_recon_nonzeros)[0], [length-1])))
            
            diff_indices = (diff_am_recon_nonzeros <= step).astype(int)
            
            diff_indices_diff = diff(diff_indices)
            diff_ones = argwhere(diff_indices_diff == 1)
            diff_minus_ones = argwhere(diff_indices_diff == -1)
            
            for i, st_tith in enumerate(diff_ones):
                am_indices = transpose(am_recon_nonzeros[st_tith[0]: diff_minus_ones[i][0]+1])[0]
                am_range = arange(am_recon_nonzeros[st_tith[0]], am_recon_nonzeros[diff_minus_ones[i][0]]+1)
                
                am_recon[am_range, k] = interp1d(am_indices, am_recon[am_indices, k])(am_range)
                
                if len(am_indices) >= min_interp_size:
                    fm_recon[am_range, k] = interp1d(am_indices, fm_recon[am_indices, k], kind=3)(am_range)
                else:
                    am_indices_tmp = concatenate((arange(0, (min_interp_size-len(am_indices))*step, step), am_indices)) 
                    
                    fm_recon[am_range, k] = interp1d(am_indices_tmp, fm_recon[am_indices_tmp, k], kind=3)(am_range)
                
                ph_recon[am_range, k]= phase_integr_interpolation(2*pi/fs*fm_recon[:, k], ph_recon[:, k], am_indices)
                
                fm_current[am_range, k] = concatenate(([fm_recon[am_range[0]][k]], fs/(2*pi)*diff(unwrap(ph_recon[am_range, k]))))
                
            if loadingScreen:
                interploop.update(1)
        
        if loadingScreen:
            interploop.close()
        
        am_current = am_recon
        
        s_recon_tmp = a0_recon + 2*multiply(am_recon, cos(ph_recon)).sum(axis=1)
        s_recon_tmpT = transpose1dArray(s_recon_tmp)
        
        SRER.append(20*log10(std_det/std(deterministic_part-s_recon_tmpT)))
        
        if printPrompts:
            print('\nSRER: {} dB in Adaptation No: {}'.format(SRER[a], a))
            print('Adaptation Time: {}\n'.format(strftime("%H:%M:%S", gmtime(time() - adptStartTime))))
        
        if a != 0:
            if SRER[a] <= SRER[a-1]:
                break
        s_recon = deepcopy(s_recon_tmp)
        
        a0_fin = a0_recon
        am_fin = am_recon
        fm_fin = fm_recon
        pm_fin = ph_recon
        
    for i, d in enumerate(DetComponents):
        if d.isVoiced:
            ti = d.ti
            am_nonzeros = argwhere(am_fin[ti])
            DetComponents[i].a0 = a0_fin[ti]
            DetComponents[i].amplitudes = arrayByIndex(am_nonzeros, am_fin[ti, am_nonzeros])
            DetComponents[i].frange = arrayByIndex(am_nonzeros, fm_fin[ti, am_nonzeros])
            DetComponents[i].pk = arrayByIndex(am_nonzeros, pm_fin[ti, am_nonzeros])
            
    endTime = time() - startTime
    if printPrompts:        
        print('Signal adapted to {} dB SRER'.format(round(max(SRER), 6)))
        print('Total Time: {}\n\n'.format(strftime("%H:%M:%S", gmtime(endTime))))
    
    if extraInfo:
        return s_recon, DetComponents, SRER, Kmax, Fmax, endTime
    
    return s_recon, DetComponents

def iqhmLS_complexamps(s, f0range, window, fs: int, iterates: int = 0):
    '''
    Computes iteratively the parameters of first order complex polynomial
    model using Least Squares. 

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    f0range : array_like
        The estimated frequencies.
    window : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.
    iterates : int, optional
        The number of iterations. The default is 0.

    Returns
    -------
    amplitudes : array_like
        Amplitude of harmonics.
    slopes : array_like
        Slope of harmonics.
    fmismatch : array_like
        Frequency mismatch.

    '''
    windowT = transpose1dArray(window)
    
    midlen = (len(s)-1)/2
    
    K = len(f0range)
    
    window_range = arange(-midlen,midlen+1)
    window_rangeT = transpose1dArray(window_range)
    
    amplitudes = zeros(K, float)
    slopes = zeros(K, float)
    fmismatch = zeros(K, float)
    for i in range(iterates+1):
        if i != 0:
            fmismatch += fs/(2*pi)*((imag(slopes)*real(amplitudes) - imag(amplitudes)*real(slopes))/abs(amplitudes)**2)
            
            diff_f0range = diff([-fs/2, f0range, fs/2])/2
            indices = argwhere(fmismatch < -diff_f0range[0:K] or fmismatch > diff_f0range[1:len(diff_f0range)])
            fmismatch = arrayByIndex(indices, 0)
        
        t = (window_rangeT*2*pi*f0range)/fs 
        E = cos(t) + 1j* sin(t)
        E = concatenate((E, tile(window_rangeT, (1, K))*E), axis=1)
        
        Ewindow = multiply(tile(windowT, (1, 2*K)), E)
        EwindowT = conjugate(transpose(Ewindow))
        R = dot(EwindowT, Ewindow) 

        #assert(cond(R) < 10**(10)),'CAUTION!!! Bad condition of matrix.'
        
        windowSignal = multiply(windowT, s)
        arr = dot(EwindowT, windowSignal)
        ampsl = dot(inv(R), arr)
            
        amplitudes = ampsl[0:K]
        slopes = ampsl[K:2*K+1]
    
    return amplitudes, slopes, fmismatch
    
def aqhmLS_complexamps(s, fm, window, fs):
    '''
    Computes the parameters of first order complex polynomial
    model using Least Squares and a FM model for the frequency. 

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    fm : array_like
        The estimated instantaneous frequencies.
    window : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.

    Returns
    -------
    amplitudes : array_like
        Amplitude of harmonics.
    slopes : array_like
        Slope of harmonics.

    '''
    windowT = transpose1dArray(window)
    
    length = len(fm)
    K = len(fm[0])
    
    midlen = int((length-1)/2)
    
    window_range = arange(-midlen,midlen+1)
    window_rangeT = transpose1dArray(window_range)
    
    f_an = zeros((K, length), float)
    for k in range(K):
        f_an[k] = lfilter([1], [1, -1], fm[:, k])
        f_an[k] -= f_an[k][midlen]
    
    t = (2*pi*f_an)/fs
    tT = transpose(t)
    
    E1 = cos(tT) + 1j* sin(tT)
    E = concatenate((E1, tile(window_rangeT, (1, K))*E1), axis=1)
    
    Ewindow = multiply(E, tile(windowT, (1, 2*K))) 
    EwindowT = conjugate(transpose(Ewindow))
    
    R = dot(EwindowT, Ewindow)
    
    #assert(cond(R) < 10**(10)),'CAUTION!!! Bad condition of matrix.'
    
    windowSignal = multiply(windowT, s)
    arr = dot(EwindowT, windowSignal)
    ampsl = dot(inv(R), arr)
    
    amplitudes = ampsl[0:K]
    slopes = ampsl[K:2*K+1]
    
    return amplitudes, slopes

def eaqhmLS_complexamps(s, am, fm, window, fs):
    '''
    Computes the parameters of first order complex polynomial
    model using Least Squares, a AM and a FM model for the frequency 

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    am : array_like
        The estimated instantaneous amplitudes.
    fm : array_like
        The estimated instantaneous frequencies.
    window : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.

    Returns
    -------
    amplitudes : array_like
        Amplitude of harmonics.
    slopes : array_like
        Slope of harmonics.

    '''
    windowT = transpose1dArray(window)
    
    length = len(fm)
    K = len(fm[0])
    
    midlen = int((length-1)/2)
    
    window_range = arange(-midlen,midlen+1)
    window_rangeT = transpose1dArray(window_range)
    
    f_an = zeros((K, length), float)
    for k in range(K):
        f_an[k] = lfilter([1], [1, -1], fm[:, k])
        f_an[k] -= f_an[k][midlen]
    
    t = (2*pi*f_an)/fs
    tT = transpose(t)
    
    E1 = cos(tT) + 1j* sin(tT)
    eps = 10e-5
    E2 = multiply(divide(eps+am, tile(am[midlen], (2*midlen+1, 1))+eps), E1)
    E = concatenate((E2, tile(window_rangeT, (1, K))*E2), axis=1)
    
    Ewindow = multiply(E, tile(windowT, (1, 2*K))) 
    EwindowT = conjugate(transpose(Ewindow))
    
    R = dot(EwindowT, Ewindow)
    
    #assert(cond(R) < 10**(15)),'CAUTION!!! Bad condition of matrix.'
    
    windowSignal = multiply(windowT, s)
    arr = dot(EwindowT, windowSignal)
    ampsl = dot(inv(R), arr)
    
    amplitudes = ampsl[0:K]
    slopes = ampsl[K:2*K+1]
    
    return amplitudes, slopes

def phase_integr_interpolation(fm_recon, ph_recon, indices):
    '''
    Computes phase interpolation using integration of instantaneous frequency.

    Parameters
    ----------
    fm_recon : array_like
        The instantaneous frequencies.
    ph_recon : array_like
        The instantaneous phases.
    indices : array_like
        The indices to be interpolated.

    Returns
    -------
    pm_final : array_like
        A simplified array-like object.

    '''
    length = len(fm_recon)
    
    pm_final = zeros(length, float)
    
    for i in range(len(indices)-1):
        pm_inst = lfilter([1], [1, -1], fm_recon[indices[i]:indices[i+1]+1])
        pm_inst += tile(ph_recon[indices[i]]-pm_inst[0], len(pm_inst))
        
        M = round((end(pm_inst) - ph_recon[indices[i+1]])/(2*pi))
        er = pi*(end(pm_inst)-ph_recon[indices[i+1]]-2*pi*M)/(2*(indices[i+1]-indices[i]))
        t = arange(0, indices[i+1]-indices[i]+1)
        ft = sin(pi*t/(indices[i+1]-indices[i]))
        ftT = transpose(ft)
        pm_inst -= lfilter([1], [1, -1], ftT*er)
        
        pm_final[indices[i]:indices[i+1]+1] = pm_inst
    
    pm_final = pm_final[indices[0]:end(indices)+1]

    return pm_final

def voicedUnvoicedFrames(s, fs: int, gender: str):
    '''
    Estimation of speech/nonspeech and voiced/unvoiced frames.

    Parameters
    ----------
    s : array_like
        The signal to be estimated.
    fs : int
        The sampling frequency.
    gender : str
        The gender of the speaker.

    Returns
    -------
    frames : array_like
        An array of structures containing the time instants, if they are voiced and if they are speech.
    frame_step : int
        The step of the frames.

    '''
    s = ellipFilter(transpose(s)[0], fs, 30)
    
    length = len(s)
    
    s_speechNonspeech_thres = -60
    v_voicedUnvoiced_thres = 10
    s_smoothedSpeech_thres = -50
    
    if gender == "male":
        s_smooth = ellipFilter(s, fs, 1000, 'lowpass')
    else:
        s_smooth = ellipFilter(s, fs, 1500, 'lowpass')
    
    windowLen = int(round(0.03*fs))
    
    if windowLen % 2 == 0:
        windowLen += 1
    
    step = int(round(0.005*fs))
    
    midlen = (windowLen-1)/2
    window_range = arange(-midlen-1, midlen, dtype=int)
    
    ti = arange(1,length,step)
    No_ti = len(ti)
    isSpeech = zeros(No_ti, dtype=bool)
    isVoiced = zeros(No_ti, dtype=bool)
    
    for i, tith in enumerate(ti):
        if tith > midlen and tith < length-midlen:
            spEn = 20*log10(std(s[tith + window_range]))
            spEn_smooth = 20*log10(std(s_smooth[tith + window_range]))
            
            isSpeech[i] = (spEn > s_speechNonspeech_thres).any()
            if isSpeech[i]:
                isVoiced[i] = (spEn-spEn_smooth < v_voicedUnvoiced_thres).any() and (spEn_smooth > s_smoothedSpeech_thres).any() 
    
    isSpeech = medfilt(isSpeech, 5)
    isVoiced = medfilt(isVoiced, 5)
    
    frames = []
    for i, tith in enumerate(ti):
        frames.append(Frame(tith, isSpeech[i], isVoiced[i]))
    
    return frames,  frames[1].ti-frames[0].ti

def getLinear(v, t):
    '''
    Linearly interpolates a time-data array
    '''
    if isContainer(t):
        value = ones((len(t),len(v[0])))
        for ind in range(len(t)):
            value[ind][0] = t[ind]
            value[ind][1:] = getLinear(v, t[ind])
    elif isinstance(t, float):
        times = v[:, 0]
        
        previ = end(argwhere(times<=t))
        
        if isEmpty(previ):
            value = v[0, 1:]
            previ = 0
            nexti = 0
            g = 1
        elif previ == len(v):
            value = v[len(v)-1, 1:]
            nexti = previ
            g = 0
        elif times[previ] == t:
            value = v[previ, 1:]
            nexti = previ
            g = 0
        else:
            nexti=previ+1

            g = (t-times[previ])/(times[nexti]-times[previ])

            if g<0 or g>1:
                raise ValueError('linearity factor unbound, g not in [0, 1]')
            
            value = v[previ, 1:]*(1-g) + v[nexti, 1:]*g;
    return value

