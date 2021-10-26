# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:00:19 2021

@author: Panagiotis Antivasis
"""
from time import time, strftime, gmtime

from structs import Deterministic, Various
from numpy import arange, zeros, blackman, hamming, \
argwhere, insert, flipud, fliplr, asarray, append, multiply, \
real, imag, pi, divide, log10, log2, angle, diff, unwrap, sin, cos, \
std, concatenate, tile, dot, ndarray, transpose, conjugate, ones, \
ceil, inf, cumsum, fix, random, sqrt, float64
#from math import log2

from numpy.linalg import inv, norm
#from numpy.linalg import cond

from scipy.interpolate import interp1d
from scipy.signal import lfilter
from scipy.io.wavfile import read

from misc import arrayByIndex, mytranspose, end, transpose1dArray, normalize, \
isContainer, isEmpty, erbs2hz, hz2erbs, primes, apply, singlelize, ellipFilter, \
loadParameters, loadOptions

from statistics import median

from copy import deepcopy

from tqdm import tqdm
from warnings import filterwarnings

def eaQHManalysis(speechFile: str, paramFile: str, ignoreSWIPEP: bool = True, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs Adaptive Quasi-Harmonic Analysis of Speech
    using the extended adaptive Quasi-Harmonic Model and decomposes 
    speech into AM-FM components according to that model.
    
    ----INPUT PARAMETERS----
    1) speechFile: string - The location of the mono .wav file to be analysed, 
    and determines the values of s, fs, len and deterministic_part.
    
    2) paramFile: string - The location of a parameter file
    containing all necessary parameters for the function. 
    The file must contain the following parameters:
        ---- gender: string - The gender of the speaker.
        ---- step: int - The step size of the processing in samples. If not contained, 
            the default value is 15.
        ---- opt: array - An array with 11 integers, each one mapped to a respective option parameter:
                opt[0]: fullWaveform, int(flag) - Full waveform length analysis.
                opt[1]: <don't care>
                opt[2]: fullBand, int(flag) - Full band analysis-in-voiced-frames flag.
                opt[3]: extended_aQHM, int(flag) - Extended aQHM.
                opt[4]: <don't care>
                opt[5]: <don't care>
                opt[6]: highPassFilter, int(flag) - High pass filtering at 30 Hz (preprocess).
                opt[7]: <don't care>
                opt[8]: SWIPEP, int(flag) - SWIPEP pitch estimator.
                opt[9]: <don't care>
                opt[10]: numPartials, int - The number of partials.
        ---- adpt: int (optional) - The maximum number of adaptations allowed. 
            If not contained, the default value is 6
        ---- NoP: int (optional) - The number of analysis window size, in pitch periods. 
            If not contained, the default value is 3
        ---- opt_pitch_f0min: float - The minimum amount of optimal estimated frequency f0.
        ---- f0sin: float (optional)- The f0 estimates for every time instant along with the time instants
            and their strength. If opt[8] == 1, this parameter is ignored. 
            If not contained, a SWIPEP pitch estimation is applied.
        ---- PAW: int - The sample of the pitch analysis window, where the analysis starts. 
            If not contained, the default value is 32.
    
    3) ignoreSWIPEP: bool (optional) - Determines if the SWIPEP pitch estimator flag (opt[8]) will be ignored or not.
    If not given, the default value is True.
    
    4) printPrompts: bool (optional) - Determines if prompts of this process will be printed. 
    If not given, the default value is True.

    5) loadingScreen: bool (optional) - Determines if a tqdm loading screen will be displayed in the console. 
    If not given, the default value is True.
    
    ----OUTPUT PARAMETERS----
    1) D: array [1 x No_ti] - The Deterministic part of the signal. An array containing elements of Deterministic-class type.
    
    2) S: array [1 x No_ti] - The Stochastic part of the signal. An array containing elements of Stochastic-class type. 
    If fullWaveform == 1, an empty array is returned.
    
    3) V: Various - Various important data.
    
    4) SRER: array [1 x adpt+1] - An array containing all the adaptation numbers of the signal.
    
    5) aSNR: array [adpt x No_ti] - An array containing each SNR (Signal to Noise Ratio) of each time instant per adaptation.
    '''
    startTime = time()
    filterwarnings("ignore")
    min_interp_size = 4
    
    parameters = loadParameters(paramFile)
    options = loadOptions(paramFile, ignoreSWIPEP)
    
    fs, s = read(speechFile)
    s = transpose1dArray(s/normalize)
    deterministic_part = s
    length = len(s)
    
    step = parameters["step"]
    PAW = parameters["PAW"]
    gender = parameters["gender"]
    maxAdpt = parameters["adpt"]
    opt_pitch_f0min = parameters["opt_pitch_f0min"]
    NoP = parameters["NoP"]
    
    if printPrompts:
        print('extended Adaptive Quasi-Harmonic Analysis initates\n')
        
        if options['extended_aQHM']:
            print('Full adaptation is applied.')
        else:
            print('Phase adaptation is applied.')
    
    if options['highPassFilter']:
        if printPrompts:
            print('High pass filtering at 30 Hz is applied.')
        s = transpose(ellipFilter(transpose(s), fs, 30))
        
    s2 = deepcopy(s)
    if options['SWIPEP'] or (not 'f0sin' in parameters):
        if printPrompts:
            print('SWIPEP pitch estimation is applied.')

        if parameters['gender'] == 'male':
            f0min = 70
            f0max = 220
        else:
            f0min = 120
            f0max = 350
            
        opt_swipep = {
                "dt": 0.001,
                "plim": [f0min, f0max],
                "dlog2p": 1/48,
                "dERBs": 0.1,
                "woverlap": 0.5,
                "sTHR": -inf
            }
        f0s = swipep(transpose(s2)[0], fs, speechFile, opt_swipep, printPrompts, loadingScreen)
        f0s[:, 1] = smooth(f0s[:, 0], f0s[:, 1])
        opt_pitch_f0min = f0min        

        f0sin = getLinear(f0s, arange(0, len(s2)-1, round(fs*5/1000))/fs)
    else:
        if printPrompts:
            print('Pitch estimations are received from the parameter file')
        f0sin = parameters["f0sin"]
        
       
    if options['fullBand']:
        if printPrompts:
            print('Full band analysis-in-voiced-frames is performed')
        Fmax = int(fs/2-200)
        
        if options['numPartials'] > 0:
            Kmax = options['numPartials']
        else:
            Kmax = int(round(Fmax/min(f0sin[:,1])) + 10)
    else:
        Fmax = int(fs/2-2000)
        Kmax = int(round(Fmax/min(f0sin[:,1])) + 10) 
    
    
    pawsample = PAW*step
    
    P, p_step = voicedUnvoicedFrames(s, fs, gender)

    if not options['fullWaveform']:
        if printPrompts:
            print('Only voiced parts analysis is performed. Noise will be added to model unvoiced speech.')
        sp_i = []
        ss = zeros((len(s), 1))
        for p in P:
            if p.isSpeech and p.isVoiced:
                sp_i = concatenate((sp_i, p.ti))
            else:
                if not isEmpty(sp_i):
                    sp_v = arange(sp_i[0]-p_step, end(sp_i)+p_step+1)
                    ss[sp_v] = s[sp_v]
                    sp_i = []
        deterministic_part = ss 
    else:
        if printPrompts:
            print('Full waveform length analysis is performed.')
        for i, p in enumerate(P):
            if p.ti > pawsample/2 and p.isSpeech and (not p.isVoiced) and p.ti < length - pawsample/2:
                P[i].isVoiced = True
            if p.ti > pawsample/2 and (not p.isSpeech) and (not p.isVoiced) and p.ti < length - pawsample/2:
                P[i].isSpeech = True
                P[i].isVoiced = True
        deterministic_part = s
        
    if printPrompts:
        print('Filename: {}'.format(speechFile))
        print('Duration: {} sec'.format(round(length/fs)))
        print('Maximum Voiced Frequency: {} Hz'.format(Fmax))
        print('Maximum Partials: {}'.format(Kmax))
        print('Analysis step: {} samples ({} sec)\n'.format(step, step/fs))
        print('Maximum number of adaptations: {}'.format(maxAdpt))
        print('Gender: {}\n'.format(gender))

    ti = arange(1,length,step)
    No_ti = len(ti)

    D = [Deterministic() for _ in range(No_ti)]
    V = Various(s=s, fs=fs, fullBand=options["fullBand"], Kmax=Kmax, Fmax=Fmax, filename=speechFile, fullWaveform=options["fullWaveform"])
    aSNR = zeros((maxAdpt, No_ti), float)
    SRER = zeros(maxAdpt+1, float)
    
    
    p = (ti/p_step)
    pf = p.astype(int)
    N = zeros(No_ti, int)

    f0_val = zeros(length, float)
    fm_cur = zeros((length, Kmax), float)
    am_cur = zeros((length, Kmax), float)
    std_det = std(deterministic_part)
    
    for m in range(maxAdpt+1):
        adptStartTime = time()
        if printPrompts:
            print('---- Adaptation No. {} ----\n'.format(m))
            
        a0_hat = zeros(length, float)
        am_hat = zeros((length, Kmax), float)
        fm_hat = zeros((length, Kmax), float)
        pm_hat = zeros((length, Kmax), float)
        
        if loadingScreen:
            analysisloop = tqdm(total=No_ti, position=0, leave=True)
            
        for i, tith in enumerate(ti):
            if loadingScreen:
                analysisloop.set_description("Analysis".format(i))
            
            if tith > pawsample and tith < length-pawsample:
                if P[pf[i]-1].isVoiced and P[pf[i]].isVoiced: 
                    if m == 0:
                        lamda = p[i] - pf[i]
                        
                        f0 = (1-lamda)*f0sin[pf[i]-1][1] + lamda*f0sin[pf[i]][1]
                        
                        K = int(min(Kmax, int(Fmax/f0)))
                        
                        fk = arange(-K,K+1)*f0
                        
                        N[i] = max(120, round((NoP/2)*(fs/f0))) 
                        
                        n = arange(-N[i]-1,N[i]) 
                        win = blackman(2*N[i]+1)
                        
                        ak, bk, df, aSNR[m][i] = iqhmLS_complexamps(s[n + tith], fk, win, fs)
                        
                        f0_val[tith-1] = f0
                    else:
                        n = arange(-N[i]-1,N[i])
                        win = hamming(2*N[i]+1)
                            
                        idx = argwhere(fm_cur[tith-1])
                        
                        if len(idx) == 0:
                            idx = 0
                            
                            fm_cur[tith-1, idx] = 140
                            am_cur[tith-1, idx] = 10e-4
                            
                            fm_tmp = fm_cur[tith + n, idx]
                            am_tmp = am_cur[tith + n, idx]
                            Kend = 1
                            
                            K = 1
                            L = len(fm_tmp)
                                
                            zr = argwhere(fm_tmp == 0)
                            nzr = argwhere(fm_tmp)
                            
                            if len(zr) != 0:
                                zridx = zr[0][0]
                                nzridx = nzr[0][0]
                                if zridx == 0:
                                    fm_tmp[zridx]= fm_tmp[nzridx]
                                    am_tmp[zridx] = am_tmp[nzridx]
                                   
                                    nzr = insert(nzr, 0, zridx)
                                
                                zridx = end(zr)
                                nzridx = end(nzr)
                                if zridx == L-1:
                                    fm_tmp[zridx] = fm_tmp[nzridx]
                                    am_tmp[zridx] = am_tmp[nzridx]
                                   
                                    nzr = append(nzr, zridx)
                            
                            x_new = arange(0, L)
                            if len(transpose(nzr)) == 1:
                                nzr = transpose(nzr)[0]
                                
                            fm_tmp = interp1d(nzr, fm_tmp[nzr])(x_new)
                            am_tmp = interp1d(nzr, am_tmp[nzr])(x_new)
                        else:
                            fm_tmp = transpose(fm_cur[tith + n, idx])
                            am_tmp = transpose(am_cur[tith + n, idx])
                            Kend = len(idx)
                        
                            K = end(idx)+1
                        
                            for k in range(Kend):
                                L = len(fm_tmp[:, k])
                                
                                zr = argwhere(fm_tmp[:, k] == 0)
                                nzr = argwhere(fm_tmp[:, k])
                                
                                if len(zr) != 0:
                                    zridx = zr[0][0]
                                    nzridx = nzr[0][0]
                                    if zridx == 0:
                                        fm_tmp[zridx][k]= fm_tmp[nzridx][k]
                                        am_tmp[zridx][k] = am_tmp[nzridx][k]
                                       
                                        nzr = insert(nzr, 0, zridx)
                                    
                                    zridx = end(zr)
                                    nzridx = end(nzr)
                                    if zridx == L-1:
                                        fm_tmp[zridx][k] = fm_tmp[nzridx][k]
                                        am_tmp[zridx][k] = am_tmp[nzridx][k]
                                       
                                        nzr = append(nzr, zridx)
                                
                                x_new = arange(0, L)
                                if len(transpose(nzr)) == 1:
                                    nzr = transpose(nzr)[0]
                                    
                                fm_tmp[:, k] = interp1d(nzr, fm_tmp[nzr, k])(x_new)
                                am_tmp[:, k] = interp1d(nzr, am_tmp[nzr, k])(x_new)
                        
                        if isinstance(idx, ndarray):
                            idx = concatenate((transpose(-flipud(idx+1))[0], [0], transpose(idx)[0]+1)) + K
                            tmp_zeros = zeros((L, 1), float)
                        
                            fm_tmp = concatenate((-flipud(fm_tmp), tmp_zeros, fm_tmp), axis=1)
                            am_tmp = concatenate((flipud(am_tmp), tmp_zeros, am_tmp), axis=1)
                        else:
                            idx = asarray([K-1, K, K+1])
                        
                            tmp_zeros = zeros((L, 1), float)
                        
                            fm_tmp = concatenate((-flipud(transpose1dArray(fm_tmp)), tmp_zeros, transpose1dArray(fm_tmp)), axis=1)
                            am_tmp = concatenate((flipud(transpose1dArray(am_tmp)), tmp_zeros, transpose1dArray(am_tmp)), axis=1)
                        
                        if not options["extended_aQHM"]:
                            ak_tmp, bk_tmp, aSNR[m][i] = aqhmLS_complexamps(s[n + tith], fm_tmp, win, fs)
                        else:
                            #am_tmp = concatenate((flipud(am_tmp), tmp_zeros, am_tmp), axis=1)
                            
                            ak_tmp, bk_tmp, aSNR[m][i] = eaqhmLS_complexamps(s[n + tith], am_tmp, fm_tmp, win, fs)
                        
                        df_tmp = fs/(2*pi)*divide(multiply(real(ak_tmp), imag(bk_tmp)) - multiply(imag(ak_tmp), real(bk_tmp)), abs(ak_tmp)**2)
                        
                        ak = arrayByIndex(idx, transpose(ak_tmp)[0])
                        bk = arrayByIndex(idx, transpose(bk_tmp)[0])
                        df = arrayByIndex(idx, transpose(df_tmp)[0])
                        
                    a0_hat[tith-1] = real(ak[K])
                    
                    ak = mytranspose(ak[K+1:2*K+1])
                    bk = mytranspose(bk[K+1:2*K+1])
                    df = mytranspose(df[K+1:2*K+1])
                    
                    ak_log_max = 20*log10(max(abs(ak)))-150
                    h = f0/(m+1)
                    
                    for k in range(K):
                        ak_log = 20*log10(abs(ak[k]))
                        
                        if ak_log > ak_log_max and abs(df[k]) < h:
                            am_hat[tith-1][k] = abs(ak[k])
                            pm_hat[tith-1][k] = angle(ak[k])
                                
                            if m == 0:
                                fm_hat[tith-1][k] = (k+1)*f0 + df[k]
                            elif f0 > opt_pitch_f0min:
                                fm_hat[tith-1][k] = fm_cur[tith-1][k] + df[k]
                            else:
                                fm_hat[tith-1][k] = fm_cur[tith-1][k]
                    D[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=True)
                else:
                   D[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=False)
            else:
                D[i] = Deterministic(ti=tith-1, isSpeech=False, isVoiced=False)
            
            if loadingScreen:
                analysisloop.update(1)
        
        if loadingScreen:
            analysisloop.close()
        
        fm_cur = zeros((length, Kmax), float)
        am_cur = zeros((length, Kmax), float)
        
        a0_hat = interp1d(ti-1, a0_hat[ti-1], kind=3, fill_value="extrapolate")(arange(0, length))

        
        if loadingScreen:
            interploop = tqdm(total=Kmax, position=0, leave=True)
            
        for k in range(Kmax): 
            if loadingScreen:
                interploop.set_description("Interpolation".format(k))
            
            nzv = argwhere(am_hat[:, k]) 
            
            dnzv = diff(concatenate(([0], transpose(nzv)[0], [length-1])))
            
            dnzv_idx = (dnzv <= step).astype(int)
            
            dnzv_idx_diff = diff(dnzv_idx)
            st_ti = argwhere(dnzv_idx_diff == 1)
            en_ti = argwhere(dnzv_idx_diff == -1)
            
            for i, st_tith in enumerate(st_ti):
                idx1 = transpose(nzv[st_tith[0]: en_ti[i][0]+1])[0]
                idx2 = arange(nzv[st_tith[0]], nzv[en_ti[i][0]]+1)
                
                am_hat[idx2, k] = interp1d(idx1, am_hat[idx1, k])(idx2)
                
                if len(idx1) >= min_interp_size:
                    fm_hat[idx2, k] = interp1d(idx1, fm_hat[idx1, k], kind=3)(idx2)
                else:
                    idx1_tmp = concatenate((arange(0, (min_interp_size-len(idx1))*step, step), idx1)) 
                    
                    #assert(len(idx1_tmp) == min_interp_size)
                    fm_hat[idx2, k] = interp1d(idx1_tmp, fm_hat[idx1_tmp, k], kind=3)(idx2)
                
                pm_hat[idx2, k]= phase_integr_interpolation(2*pi/fs*fm_hat[:, k], pm_hat[:, k], idx1)
                
                fm_cur[idx2, k] = concatenate(([fm_hat[idx2[0]][k]], fs/(2*pi)*diff(unwrap(pm_hat[idx2, k]))))
                
            if loadingScreen:
                interploop.update(1)
        
        if loadingScreen:
            interploop.close()
        
        am_cur = am_hat
        
        s_hat = a0_hat + 2*multiply(am_hat, cos(pm_hat)).sum(axis=1)
        s_hatT = transpose1dArray(s_hat)
        
        SRER[m] = 20*log10(std_det/std(deterministic_part-s_hatT))
        
        if printPrompts:
            print('\nSRER: {} dB in Adaptation No: {}'.format(SRER[m], m))
            print('Adaptation Time: {}\n'.format(strftime("%H:%M:%S", gmtime(time() - adptStartTime))))
        
        if m != 0:
            if SRER[m] <= SRER[m-1]:
                break
            V.qh = s_hat
        a0_fin = a0_hat
        am_fin = am_hat
        fm_fin = fm_hat
        pm_fin = pm_hat
        
    for i, d in enumerate(D):
        if d.isVoiced:
            ti = d.ti
            idx = argwhere(am_fin[ti])
            D[i].a0 = a0_fin[ti]
            D[i].ak = arrayByIndex(idx, am_fin[ti, idx])
            D[i].fk = arrayByIndex(idx, fm_fin[ti, idx])
            D[i].pk = arrayByIndex(idx, pm_fin[ti, idx])
            
    S = []
    if not options['fullWaveform']:
        #----NOT TESTED----
        try:
            from scikits.talkbox import lpc 
            from numpy.fft import fft
            from scipy.signal import filtfilt
            from structs import Stochastic
            
            S = [Stochastic() for _ in range(No_ti)]
            
            s_noi = s - s_hatT
            
            s_noi = ellipFilter(s, fs, 1500)
            
            M = 25
            s_env = maxfilt(abs(s), 9)
        
            s_noi_env = filtfilt(ones(M)/M, 1, abs(s_noi));    
            
            V.noi_env = s_noi_env
            
            step = 5*fs/1000
            winLen1 = 30*fs/1000+1
            N1 = (winLen1-1)/2
            n1 = arange(-N1, N1+1)
            winLen2 = 25*fs/1000+1
            N2 = (winLen2-1)/2
            n2 = arange(-N2, N2+1)
            win2 = hamming(winLen2)
            nf_win2 = sum(win2)
            
            LPCord = 20
            NFFT = 1024
            NoS = 4
            
            ti = arange(1,length,step)
            No_ti = len(ti)
            
            if loadingScreen:
                stochloop = tqdm(total=No_ti, position=0, leave=True)
            for i, tith in enumerate(ti):
                if loadingScreen:
                    stochloop.set_description("Stochastic Part".format(i))
                if tith > N1 and tith < length-N1:
                    p_i = (tith)/p_step
                    
                    pf_i = int(p_i)
                    
                    if P[pf_i-1].isSpeech and P[pf_i].isSpeech:
                        if (not P[pf_i-1].isVoiced) and (not P[pf_i].isVoiced):
                            noi_tmp = s[tith + n1 - 1]
                            s_env_tmp = s_env[tith + n2 - 1]
                        else:
                            noi_tmp = s_noi[tith + n1 - 1]
                            s_env_tmp = s_noi_env[tith + n2 - 1]
                        
                        ap, g = lpc(noi_tmp, LPCord)
                        
                        s_env_tmp = multiply(s_env_tmp, win2)
                        S_env = fft(concatenate((s_env_tmp[N2:], zeros((NFFT-winLen2, 1)), s_env_tmp[0:N2-1])), NFFT)/nf_win2
                        am_s, fr_s = peak_picking(abs(S_env[0:NFFT/2]))
                        idx = sorted(am_s, reverse=True)
                        am_s = concatenate(([S_env[1]/2], S_env[fr_s[idx[0:min(NoS, len(idx))]]]))
                        fm_s = concatenate(([0], fr_s[idx[0:min(NoS,len(idx))]]-1))*fs/NFFT
                        
                        S[i] = Stochastic(ti=tith-1, isSpeech=True, ap = ap, env_ak = abs(am_s), env_fk = fm_s, env_pk = angle(am_s))
                    else:
                        S[i] = Stochastic(ti=tith-1, isSpeech=False)
                else:
                    S[i] = Stochastic(ti=tith-1, isSpeech=False)
                if loadingScreen:
                    stochloop.update(1)
        
            if loadingScreen:
                stochloop.close()
        except ModuleNotFoundError:
            print("Stochastic part skipped: 'pip install scikits' is required")
    endTime = strftime("%H:%M:%S", gmtime(time() - startTime))
    if printPrompts:        
        print('Signal adapted to {} dB SRER'.format(round(max(SRER), 6)))
        #endTime = strftime("%H:%M:%S", gmtime(time() - startTime))
        print('Total Time: {}\n\n'.format(strftime("%H:%M:%S", gmtime(time() - startTime))))
    
    return round(max(SRER), 6), endTime
    #return D, S, V, SRER, aSNR

def eaQHMsynthesis(D, S, V, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs speech synthesis using the extended adaptive Quasi-Harmonic
    Model parameters from eaQHManalysis and resynthesizes speech from 
    its AM-FM components according to the eaQHM model.
    
    ----INPUT PARAMETERS----
    1) D: array [1 x No_ti] - The Deterministic part of the signal. An array containing elements of Deterministic-class type.
    2) S: array [1 x No_ti] - The Stochastic part of the signal. An array containing elements of Stochastic-class type. 
    3) V: Various - Various important data.
    4) printPrompts: bool (optional) - Determines if prompts of this process will be printed. 
    If not given, the default value is True.
    5) loadingScreen: bool (optional) - Determines if a tqdm loading screen will be displayed in the console. 
    If not given, the default value is True.
    ----OUTPUT PARAMETERS----
    1) s - The reconstructed speech signal
    
    2) qh - The reconstructed deterministic part (quasi-harmonic)
    
    3) noi - The reconstructed stochastic part. 
    If len(S) == 0,  an empty array is returned.
    '''
    startTime = time()
    min_interp_size = 4
    
    Kmax = V.Kmax
    len1 = D[len(D)-1].ti + 500
    fs= V.fs
    
    length = len1
    
    ti = zeros(len(D), int)
    a0 = zeros(len1, float)
    am = zeros((length, Kmax), float)
    fm = zeros((length, Kmax), float)
    pm = zeros((length, Kmax), float)
    pm_tmp = zeros((length, Kmax), float)
    am_tmp = zeros((length, Kmax), float)
    
    for i, d in enumerate(D):
        ti[i] = d.ti
        if d.isVoiced:
            a0[d.ti] = d.a0
            k = arange(0, len(d.ak))
            am[d.ti, k] = transpose(d.ak)
            fm[d.ti, k] = transpose(d.fk)
            pm[d.ti, k] = transpose(d.pk)
    
    step = ti[1] - ti[0]
    
    a_0 = interp1d(ti, a0[ti], fill_value="extrapolate")(arange(0, len1))
    qh = a_0
    
    if printPrompts:
        print('extended Adaptive Quasi-Harmonic Analysis initates\n')
        print('Performing synthesis of filename {}'.format(V.filename))
    
    if loadingScreen:
        synthesisloop = tqdm(total=Kmax, position=0, leave=True)
    
    for k in range(Kmax): 
        if loadingScreen:
            synthesisloop.set_description("Synthesis".format(k))
                
        nzv = argwhere(am[:, k]) 
        
        dnzv = diff(concatenate(([0], transpose(nzv)[0], [len1])))
        
        dnzv_idx = (dnzv <= step).astype(int)
        
        dnzv_idx_diff = diff(dnzv_idx)
        st_ti = argwhere(dnzv_idx_diff == 1)
        en_ti = argwhere(dnzv_idx_diff == -1)
        
        for i, st_tith in enumerate(st_ti):
            idx1 = transpose(nzv[st_tith[0]: en_ti[i][0]+1])[0]
            idx2 = arange(nzv[st_tith[0]], nzv[en_ti[i][0]]+1)
            
            am_tmp[idx2, k] = interp1d(idx1, am[idx1, k])(idx2)
            
            z1 = fm[idx1, k] < 0
            if z1.any():
                fm[idx1[argwhere(z1)], k] = -fm[idx1[argwhere(z1)], k]
            
            if len(idx1) >= min_interp_size:
                fm[idx2, k] = interp1d(idx1, fm[idx1, k], kind=3)(idx2)
            else:
                idx1_tmp = concatenate((arange(0, (min_interp_size-len(idx1))*step, step), idx1)) 
                
                #assert(len(idx1_tmp) == min_interp_size)
                fm[idx2, k] = interp1d(idx1_tmp, fm[idx1_tmp, k], kind=3)(idx2)
            
            pm_tmp[idx2, k]= phase_integr_interpolation(2*pi/fs*fm[:, k], pm[:, k], idx1)
            
            try:
                qh[idx2] = qh[idx2] + 2*multiply(am_tmp[idx2, k], cos(pm_tmp[idx2, k])).sum(axis=1)
            except IndexError:
                qh[idx2] = qh[idx2] + 2*multiply(am_tmp[idx2, k], cos(pm_tmp[idx2, k]))
        if loadingScreen:
            synthesisloop.update(1)
        
    if loadingScreen:
        synthesisloop.close()
        
    noi = zeros(length, float)
        
    if not V.fullBand:
        #----NOT TESTED----
        from numpy import hanning
        from random import random
        N = S[1].ti - S[0].ti
        n = arange(-N-1, N)
        win = hanning(2*N+1)
        
        for st, i in enumerate(S):
            if st.isSpeech and st.ti > N:
                e = zeros(2*N+1, float)
                
                for i in range(len(e)):
                    e[i] = 2*random()-1
                
                e = lfilter([1], st.ap, e)
                
                e_env = 2*abs(st.env_ak)*cos(2*pi*st.env_fk*n/fs + tile(st.env_pk, (1, 2*N+1)));
                e = e/std(e)*transpose(e_env);

                noi[st.ti + n] = noi[st.ti + n] + win*e;
    
    s = qh + noi
    
    print('Signal synthesised')
    print('Total Time: {}\n\n'.format(strftime("%H:%M:%S", gmtime(time() - startTime))))
    
    return s, qh, noi

def smooth(t, s, windur=0.1):
    '''
    Smooths the signal using median filtering and zero-phase filter
    
    ----INPUT PARAMETERS----
    1) t: array - The time 
    2) s: array - The signal 
    3) windur: float - Defines the order of the window
    
    ----OUTPUT PARAMETERS----
    The smoothed signal
    '''
    from scipy.signal import filtfilt
    order = round(windur/median(diff(t))/2)*2+1
    
    if order > 1:
        s = medfilt(s, int(order))
        
        medvalue = median(s)
            
        if ceil(order/2)>1:
            vals1 = s
            lenori = len(vals1)
            inds = arange(0, lenori)

            if lenori<3*order/2:
                length = int(round(2*order/2))
                vals1 = concatenate((medvalue*ones((length,1)), vals1, medvalue*ones((length,1))))
                inds = arange(length+1, len(vals1)-length+1)
            win = blackman(ceil(order/2))
            win = win/sum(win)
            vals1 = filtfilt(win,1,vals1)
            vals1 = vals1[inds]
            if lenori != len(vals1):
                raise ValueError('length mismatch: {}, {}'.format(lenori, len(vals1)))
            return vals1   
        
def iqhmLS_complexamps(s, fk, win, fs: int, iterates: int = 0):
    '''
    Computes iteratively the parameters of first order complex polynomial
    model using Least Squares. 
    
    ----INPUT PARAMETERS----
    1) s: array - The part of the signal to be computed.
    2) fk: array [1 x K] - The estimated frequencies (where the initial analysis is performed)
    3) win: array - The window of the signal to be computed.
    4) fs: int - The sampling frequency.
    5) iterates: int - The number of iterations.
    
    ----OUTPUT PARAMETERS----
    1) ak: array - Amplitude of harmonics
    2) bk: array - Slope of harmonics
    3) df: array - Frequency mismatch
    4) SNR: float - Signal-to-noise ratio
    '''
    wint = transpose1dArray(win)
    
    N = (len(s)-1)/2
    
    K = len(fk)
    
    n = arange(-N,N+1)
    nt = transpose1dArray(n)
    
    ak = zeros(K, float)
    bk = zeros(K, float)
    df = zeros(K, float)
    for i in range(iterates+1):
        if i != 0:
            df += fs/(2*pi)*((imag(bk)*real(ak) - imag(ak)*real(bk))/abs(ak)**2)
            
            cfk = diff([-fs/2, fk, fs/2])/2
            idx = argwhere(df < -cfk[0:K] or df > cfk[1:len(cfk)])
            df = arrayByIndex(idx, 0)
        
        t = (nt*2*pi*fk)/fs 
        E = cos(t) + 1j* sin(t)
        E = concatenate((E, tile(nt, (1, K))*E), axis=1)
        
        Ew = multiply(tile(wint, (1, 2*K)), E)
        Ewt = conjugate(transpose(Ew))
        R = dot(Ewt, Ew) 

        #assert(cond(R) < 10**(10)),'CAUTION!!! Bad condition of matrix.'
        
        wins = multiply(wint, s)
        arr = dot(Ewt, wins)
        x = dot(inv(R), arr)
            
        ak = x[0:K]
        bk = x[K:2*K+1]
    
    y = real(dot(E, concatenate((ak, bk))))

    SNR = 20*log10(std(s)/std(s-y))

    return ak, bk, df, SNR
    
def aqhmLS_complexamps(s, fm, win, fs):
    '''
    Computes the parameters of first order complex polynomial
    model using Least Squares and a FM model for the frequency. 
    
    ----INPUT PARAMETERS----
    1) s: array - The part of the signal to be computed.
    2) fm: Estimated inst. frequencies, where the analysis is performed.
    3) win: array - The window of the signal to be computed.
    4) fs: int - The sampling frequency.
    
    ----OUTPUT PARAMETERS----
    1) ak: array - Amplitude of harmonics
    2) bk: array - Slope of harmonics
    3) SNR: float - Signal-to-noise ratio
    '''
    
    #----NOT TESTED----
    wint = transpose1dArray(win)
    
    length = len(fm)
    K = len(fm[0])
    
    N = int((length-1)/2)
    
    n = arange(-N,N+1)
    nt = transpose1dArray(n)
    
    f_an = zeros((K, length), float)
    for k in range(K):
        f_an[k] = lfilter([1], [1, -1], fm[:, k])
        f_an[k] -= f_an[k][N]
    
    t = (2*pi*f_an)/fs
    tT = transpose(t)
    
    E1 = cos(tT) + 1j* sin(tT)
    E = concatenate((E1, tile(nt, (1, K))*E1), axis=1)
    
    Ew = multiply(E, tile(wint, (1, 2*K))) 
    Ewt = conjugate(transpose(Ew))
    
    R = dot(Ewt, Ew)
    
    #assert(cond(R) < 10**(10)),'CAUTION!!! Bad condition of matrix.'
    
    wins = multiply(wint, s)
    arr = dot(Ewt, wins)
    x = dot(inv(R), arr)
    
    ak = x[0:K]
    bk = x[K:2*K+1]
    
    y = real(dot(E, concatenate((ak, bk))))
    
    SNR = 20*log10(std(s)/std(s-y)) 

    return ak, bk, SNR

def eaqhmLS_complexamps(s, am, fm, win, fs):
    '''
    Computes the parameters of first order complex polynomial
    model using Least Squares, a AM and a FM model for the frequency. 
    
    ----INPUT PARAMETERS----
    1) s: array - The part of the signal to be computed.
    2) am: Estimated inst. amplitudes, where the analysis is performed.
    3) fm: Estimated inst. frequencies, where the analysis is performed.
    4) win: array - The window of the signal to be computed.
    5) fs: int - The sampling frequency.
    
    ----OUTPUT PARAMETERS----
    1) ak: array - Amplitude of harmonics
    2) bk: array - Slope of harmonics
    3) SNR: float - Signal-to-noise ratio
    '''
    wint = transpose1dArray(win)
    
    length = len(fm)
    K = len(fm[0])
    
    N = int((length-1)/2)
    
    n = arange(-N,N+1)
    nt = transpose1dArray(n)
    
    f_an = zeros((K, length), float)
    for k in range(K):
        f_an[k] = lfilter([1], [1, -1], fm[:, k])
        f_an[k] -= f_an[k][N]
    
    t = (2*pi*f_an)/fs
    tT = transpose(t)
    
    E1 = cos(tT) + 1j* sin(tT)
    eps = 10e-5
    E2 = multiply(divide(eps+am, tile(am[N], (2*N+1, 1))+eps), E1)
    E = concatenate((E2, tile(nt, (1, K))*E2), axis=1)
    
    Ew = multiply(E, tile(wint, (1, 2*K))) 
    Ewt = conjugate(transpose(Ew))
    
    R = dot(Ewt, Ew)
    
    #assert(cond(R) < 10**(15)),'CAUTION!!! Bad condition of matrix.'
    
    wins = multiply(wint, s)
    arr = dot(Ewt, wins)
    x = dot(inv(R), arr)
    
    ak = x[0:K]
    bk = x[K:2*K+1]
    
    y = real(dot(E, concatenate((ak, bk))))
    
    SNR = 20*log10(std(s)/std(s-y))

    return ak, bk, SNR

def phase_integr_interpolation(fm_hat, pm_hat, idx):
    '''
    Computes phase interpolation using integration of instantaneous frequency.
    
    ----INPUT PARAMETERS----
    1) fm_hat: array - The instantaneous frequency 
    2) pm_hat: array - The instantaneous phase 
    3) idx: array - The indices to be interpolated.

    ----OUTPUT PARAMETERS----
    A simplified array-like object.
    '''
    length = len(fm_hat)
    
    pm_final = zeros(length, float)
    
    for i in range(len(idx)-1):
        pm_inst = lfilter([1], [1, -1], fm_hat[idx[i]:idx[i+1]+1])
        pm_inst += tile(pm_hat[idx[i]]-pm_inst[0], len(pm_inst))
        
        M = round((end(pm_inst) - pm_hat[idx[i+1]])/(2*pi))
        er = pi*(end(pm_inst)-pm_hat[idx[i+1]]-2*pi*M)/(2*(idx[i+1]-idx[i]))
        t = arange(0, idx[i+1]-idx[i]+1)
        ft = sin(pi*t/(idx[i+1]-idx[i]))
        ftT = transpose(ft)
        pm_inst -= lfilter([1], [1, -1], ftT*er)
        
        pm_final[idx[i]:idx[i+1]+1] = pm_inst
    
    pm_final = pm_final[idx[0]:end(idx)+1]

    return pm_final

def maxfilt(x, p):
    '''
    Performs maxian filtering of order p.
    
    ----INPUT PARAMETERS----
    1) x: array - The signal 
    2) p: int - The order of the filter
    
    ----OUTPUT PARAMETERS----
    The filtered signal
    '''
    #----NOT TESTED----
    xt = transpose(x)
    L = len(xt)
    
    ad = (p-1)/2
    if ad == 0:
        return xt
    from scipy.linalg import toeplitz
    x = concatenate((x[0]*ones(ad), xt, x[L]*ones(ad)))
    
    A = fliplr(toeplitz(fliplr(x[1:L]), x[L:L+p-1]))
    
    return max(A)

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
    Amed = []
    for i in range(len(A)):
        Amed.append(median(A[i]))
    return Amed

def peak_picking(x):
    '''
    Performs peak picking on a signal.

    ----INPUT PARAMETERS----
    x: array - The signal

    ----OUTPUT PARAMENTERS----
    1) x_max: the values of the peaks
    2) x_pos: the location of the peaks (in samples) 
    '''
    #----NOT TESTED----
    end = len(x)-1
    lDiff = x[1:end-1]-x[0:end-2]
    rDiff = x[2:end]-x[1:end-1]
    
    x_pos = argwhere(lDiff > 0 and rDiff < 0)
    x_max = x[x_pos]
    return x_max, x_pos

def voicedUnvoicedFrames(s, fs, gender):
    '''
    Estimation of speech/nonspeech and voiced/unvoiced frames.
    
    ----INPUT PARAMETERS----
    1) s: array - The signal.
    2) fs: int - The sampling frequency.
    
    ----OUTPUT PARAMETERS----
    1) P: array - An array of structures containing the time instants, 
    if they are voiced and if they are speech.
    2) p_step: The step of the frames.
    '''
    from structs import Frame
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
    
    N = (windowLen-1)/2
    n = arange(-N-1, N, dtype=int)
    
    ti = arange(1,length,step)
    No_ti = len(ti)
    isSpeech = zeros(No_ti, dtype=bool)
    isVoiced = zeros(No_ti, dtype=bool)
    
    for i, tith in enumerate(ti):
        if tith > N and tith < length-N:
            spEn = 20*log10(std(s[tith + n]))
            spEn_smooth = 20*log10(std(s_smooth[tith + n]))
            
            isSpeech[i] = (spEn > s_speechNonspeech_thres).any()
            if isSpeech[i]:
                isVoiced[i] = (spEn-spEn_smooth < v_voicedUnvoiced_thres).any() and (spEn_smooth > s_smoothedSpeech_thres).any() 
    
    isSpeech = medfilt(isSpeech, 5)
    isVoiced = medfilt(isVoiced, 5)
    
    P = []
    for i, tith in enumerate(ti):
        P.append(Frame(tith, isSpeech[i], isVoiced[i]))
    
    return P,  P[1].ti-P[0].ti
    
def swipep(x, fs, speechFile, opt, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs a pitch estimation of the signal for every time instant
    
    ----INPUT PARAMETERS----
    1) x: array - The signal
    2) fs: int - The sampling frequency
    3) opt: dict - A dictionary containing all necessary parameters for the function to run.
    It must contain the following parameters:
        ---- dt: float - Estimation is performed every dt seconds
        ---- plim: array[2] - An array containing 2 numbers, the first being the minimum frequency
            where the pitch is searched and the second being the maximum frequency
        ---- dlog2p: float - The units the signal is distributed in samples on a base-2 logarithm scale. 
        ---- dERBs: float - The step size of ERBs
        ---- woverlap: float - The overlap of the Hanning window
        ---- sTHR: int - The threshold of the pitch strength
    
    ----OUTPUT PARAMETERS----
    An array containing the time instants, the estimation for each instant and the strength of each estimation.
    '''
    from numpy import hanning, power, empty, polyfit, polyval, e, nan
    from scipy.signal import spectrogram, stft
    #from matlab import specgram
    from matplotlib.pyplot import title, xlabel, ylabel
    from matplotlib.mlab import specgram
    from debug import printIfCond, compare#
    from misc import matToObject
    from misc import myHann, arrayMax, spline_interp, mySpecgram
    
    debugSwipep = False
    workspace = matToObject('../thesis_files/workspaces/swipepEnd.mat')
    
    #printIfCond(debugSwipep, 'x difference: ', compare(x, workspace['x']))
    #printIfCond(debugSwipep, 'fs difference: ', compare(fs, workspace['fs']))
    
    t = arange(0, len(x)/fs+opt['dt'], opt['dt'])
    #printIfCond(debugSwipep, 't difference: ', compare(t, workspace['t'].transpose()[0]))
    
    log2pc = arange(log2(opt['plim'][0]), log2(opt['plim'][1]), opt['dlog2p'])
    #printIfCond(debugSwipep, 'log2pc difference: ', compare(log2pc, workspace['log2pc'].transpose()[0]))
                
    pc = power(2, log2pc)
    #printIfCond(debugSwipep, 'pc difference: ', compare(pc, workspace['pc'].transpose()[0]))
    
    S = zeros((len(pc), len(t)))
    
    logWs = apply(log2(divide(8*fs, opt['plim'])), round)
    #printIfCond(debugSwipep, 'logWs difference: ', compare(logWs, workspace['logWs']))
    
    ws = power(2, arange(logWs[0], logWs[1]-1, -1))
    #printIfCond(debugSwipep, 'ws difference: ', compare(ws, workspace['ws']))
    
    pO = divide(8*fs,ws)
    #printIfCond(debugSwipep, 'pO difference: ', compare(pO, workspace['pO']))
    
    d = 1 + log2pc - log2(pO[0])
    #printIfCond(debugSwipep, 'd difference: ', compare(d, workspace['d']))
    
    fERBs = erbs2hz(arange(hz2erbs(min(pc)/4), hz2erbs(fs/2), opt['dERBs']))
    #printIfCond(debugSwipep, 'fERBs difference: ', compare(fERBs, workspace['fERBs']))
    
    if loadingScreen:
        wsloop = tqdm(total=len(ws), position=0, leave=True)
                      
    for i in range(len(ws)):
        #printIfCond(debugSwipep, "----i: ", i)
        if loadingScreen:
            wsloop.set_description("SWIPEP".format(i))
        workspaceSpec = matToObject('../thesis_files/workspaces/swipep_in_i_1.mat') 
        
        dn = int(max(1, round(8*(1-opt['woverlap'])*fs/pO[i])))
        
        xzp = concatenate((zeros((int(ws[i]/2))), x, zeros((dn + int(ws[i]/2)))))
        
        w = myHann(ws[i])
        o = int(max(0, round(ws[i] - dn)))
        
        X, f, ti = mySpecgram(xzp, int(ws[i]), fs, w, o)

        '''printIfCond(debugSwipep, 'dn difference: ', compare(dn, workspaceSpec['dn']))
        printIfCond(debugSwipep, 'xzp difference: ', compare(xzp, workspaceSpec['xzp']))
        printIfCond(debugSwipep, 'w difference: ', compare(w, workspaceSpec['w'].transpose()[0], ignoreZeros=True))
        printIfCond(debugSwipep, 'o difference: ', compare(o, workspaceSpec['o']))
        printIfCond(debugSwipep, 'f difference: ', compare(f, workspaceSpec['f']))
        printIfCond(debugSwipep, 'X difference: ', compare(X, workspaceSpec['X']))
        printIfCond(debugSwipep, 'ti difference: ', compare(ti, workspaceSpec['ti']))
        '''
        ip = i+1
        if len(ws) == 1:
            j = transpose1dArray(apply(pc, int))
            k = []
        elif i == len(ws)-1:
            #j = argwhere(d - i > 0)
            #k = argwhere(d[j] - i < 1)
            j = argwhere(d - ip > -1)
            k = argwhere(d[j] - ip < 0)
        elif i == 0:
            #j = argwhere(d - i < 2)
            #k = argwhere(d[j] - i > 1)
            
            j = argwhere(d - ip < 1)
            k = argwhere(d[j] - ip > 0)
        else:
            j = argwhere(abs(d - ip) < 1)
            k = arange(0, len(j))
        
        fERBs = fERBs[singlelize(argwhere(fERBs > pc[j[0]]/4)[0]):]
        #printIfCond(debugSwipep, 'fERBs difference: ', compare(fERBs, workspaceSpec['fERBs']))
        
        #L = interp1d(f, transpose(abs(X)), kind=3, fill_value=0)(fERBs)
        '''L0 = []
        for i in range(len(X[0])):
            L0.append(spline_interp(f, abs(X[:, i]), fERBs))
            
        L2 = sqrt(arrayMax(0, asarray(L0).transpose()))'''
        
        L = sqrt(arrayMax(0, interp1d(f, abs(X), kind=3, fill_value=0, axis=0)(fERBs)))
        '''for xi in range(len(L)):
            for xj in range(len(L[xi])):
                if L[xi][xj] < 0:
                    L[xi][xj] = 0.0#10e-20#0.0
                else:
                    L[xi][xj] = sqrt(float64(L[xi][xj]))
        #
        L[L==0] = 10e-20'''
        printIfCond(debugSwipep, 'L difference: ', compare(L, workspaceSpec['L']))
        
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j]) 
        
        if len(Si[0]) > 1:
            Si = interp1d(ti, Si, 'linear', fill_value=nan)(t)
        else:
            Si = empty((len(Si), len(t)), dtype=object)
            
        #printIfCond(debugSwipep, 'Si difference: ', compare(Si, workspaceSpec['Si']))
        
        if isContainer(k[0]):
            k = k[:, 0]
        #printIfCond(debugSwipep, 'k difference: ', compare(k, workspaceSpec['k'].transpose()[0]))
        
        lamda = d[ j[k] ] - (i+1)
        #printIfCond(debugSwipep, 'lamda difference: ', compare(lamda, workspaceSpec['lambda']))
        
        mu = ones( len(j) )
        mu[k] = 1 - abs( transpose(lamda) )
        #printIfCond(debugSwipep, 'mu difference: ', compare(mu, workspaceSpec['mu']))
        
        S[transpose(j)[0],:] += multiply(transpose(tile(mu,(len(Si[1]), 1))), Si)
        
        if loadingScreen:
            wsloop.update(1)
           
    #printIfCond(debugSwipep, "----end of loop")
    if loadingScreen:
        wsloop.close()
       
    '''printIfCond(debugSwipep, 'dn difference: ', compare(dn, workspace['dn']))
    printIfCond(debugSwipep, 'xzp difference: ', compare(xzp, workspace['xzp']))
    printIfCond(debugSwipep, 'w difference: ', compare(w, workspace['w'].transpose()[0], ignoreZeros=True))
    printIfCond(debugSwipep, 'o difference: ', compare(o, workspace['o']))
    printIfCond(debugSwipep, 'f difference: ', compare(f, workspace['f']))
    printIfCond(debugSwipep, 'ti difference: ', compare(ti, workspace['ti']))
    printIfCond(debugSwipep, 'X difference: ', compare(X, workspace['X'], ignoreZeros=True, ignoreSign=True))
    printIfCond(debugSwipep, 'fERBs difference: ', compare(fERBs, workspace['fERBs']))
    printIfCond(debugSwipep, 'L difference: ', compare(L, workspace['L'].transpose(), ignoreZeros=True))
    printIfCond(debugSwipep, 'Si difference: ', compare(Si, workspace['Si']))
    printIfCond(debugSwipep, 'lamda difference: ', compare(lamda, workspace['lambda']))
    printIfCond(debugSwipep, 'mu difference: ', compare(mu, workspace['mu']))
    '''
    p = empty((len(S[0]), 1), dtype=object)
    s = empty((len(S[0]), 1), dtype=object)
    for j in range(len(S[0])):
        s[j] = max(S[:, j])
        i = singlelize(argwhere(S[:, j] == s[j])[0])
        
        if s[j] < opt['sTHR']:
            continue
        
        if i == 0 or i == len(pc)-1:
            p[j] = pc[i]
        else:
            I = arange(i-1, i+2)
            tc = 1 / pc[I]
            ntc = (tc/tc[1] - 1) * 2*pi
            c = polyfit(ntc, S[I, j], 2)
            
            ftc_step = 1/12/100
            ftc = 1/power(2, arange(log2(pc[I[0]]), log2(pc[I[2]])+ftc_step, ftc_step))
            nftc = (ftc/tc[1] - 1) * 2*pi
            arr = polyval(c, nftc)
            s[j] = max(arr)
            k = singlelize(argwhere(arr == s[j]))
            p[j] = power(2, log2(pc[I[0]]) + k/12/100)
    
    #printIfCond(debugSwipep, 'j difference: ', compare(j, workspace['j']))
    #printIfCond(debugSwipep, 'k difference: ', compare(k, workspace['k']))
    printIfCond(debugSwipep, 't difference: ', compare(transpose1dArray(t), workspace['t']))
    printIfCond(debugSwipep, 'p difference: ', compare(p, workspace['p']))
    printIfCond(debugSwipep, 's difference: ', compare(s, workspace['s']))
    
    return concatenate((transpose1dArray(t), p, s), axis=1)

def pitchStrengthAllCandidates(f, L, pc):
    from debug import printIfCond, compare
    from misc import matToObject
    from copy import deepcopy
    
    debugAllC = False
    workspace = matToObject('../thesis_files/workspaces/allCand_in_i_1.mat')
    printIfCond(debugAllC, '----pitchStrengthAllCandidates')
    
    printIfCond(debugAllC, 'f difference: ', compare(f, workspace['f'].transpose()[0]))
    printIfCond(debugAllC, 'L difference: ', compare(L, workspace['L']))
    printIfCond(debugAllC, 'pc difference: ', compare(pc, workspace['pc']))
    
    S = zeros((len(pc), len(L[1])))
    k = zeros(len(pc)+1, dtype=int)
    for j in range(len(k)-1):
        k[j+1] = k[j] + argwhere( f[k[j]:] > pc[j]/4)[0]
        

    k = k[1:]
    printIfCond(debugAllC, 'k difference: ', compare(k, workspace['k'], ignoreZeros=True))
    
    N = sqrt(flipud(cumsum(flipud(multiply(L,L)), axis=0)))
    printIfCond(debugAllC, 'N difference: ', compare(N, workspace['N']))
    
    for j in range(len(pc)):
        n = float64(deepcopy(N[k[j], :]))
        #for ni in range(len(n)):
        #    if n[ni] == 0:
        #        n[ni] = inf
        n[n==0] = inf#10e10#inf
        NL = L[k[j]:,:] / tile( n, (len(L)-k[j], 1))
        S[j,:] = pitchStrengthOneCandidate(f[k[j]:], NL, pc[j])
    
    printIfCond(debugAllC, 'n difference: ', compare(n, workspace['n'], ignoreZeros=True))
    printIfCond(debugAllC, 'NL difference: ', compare(NL, workspace['NL'], ignoreZeros=True))
    printIfCond(debugAllC, 'S difference: ', compare(S, workspace['S']))
    
    return S

def pitchStrengthOneCandidate(f, NL, pc):
    #from debug import printIfCond, compare
    #from misc import matToObject
    
    #debugOneC = True
    #workspace = matToObject('../thesis_files/workspaces/oneCand_in_j_1.mat')
    #printIfCond(debugOneC, '----pitchStrengthOneCandidate')
    
    #printIfCond(debugOneC, 'f difference: ', compare(f, workspace['f'].transpose()[0]))
    #printIfCond(debugOneC, 'NL difference: ', compare(NL, workspace['NL'], ignoreZeros=True))
    #printIfCond(debugOneC, 'pc difference: ', compare(pc, workspace['pc']))
    
    n = int(singlelize(fix( end(f)/pc - 0.75 )))
    if n==0:
        return None
    
    #printIfCond(debugOneC, 'n difference: ', compare(n, workspace['n']))
    
    k = zeros(len(f))
    q = f / pc
    #printIfCond(debugOneC, 'q difference: ', compare(q, workspace['q']))
    
    pr = concatenate(([1], primes(n)))
    for i in pr:
        a = abs( q - i )
        p = argwhere(a < .25)
        k[p] = cos(2*pi*q[p])
        v = argwhere((0.25 < a) & (a < 0.75))
        k[v] = k[v] + cos( 2*pi * q[v] ) / 2
    k = k * apply(1/f, sqrt)
    k = k / norm( k[k>0] )
    #printIfCond(debugOneC, 'k difference: ', compare(k, workspace['k']))
    
    return dot(transpose(k), NL)

def getLinear(v, t):
    '''
    Linearly interpolates a time-data array
    
    ----INPUT PARAMETERS----
    1) v: array[len x 2] - The array containing the time-data on the first column
    and the values on the second column
    2) t: array - The time-data
    
    
    ----OUTPUT PARAMETERS----
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
                raise ValueError('linearity factor unbound, g not in [0;1]')
            
            value = v[previ, 1:]*(1-g) + v[nexti, 1:]*g;
    return value

