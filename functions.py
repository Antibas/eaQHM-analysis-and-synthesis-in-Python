# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:00:19 2021

@author: Panagiotis Antivasis
"""
from time import time, strftime, gmtime

from structs import Deterministic, Various, Frame
from numpy import arange, zeros, blackman, hamming, \
argwhere, insert, flipud, asarray, append, multiply, \
real, imag, pi, divide, log10, log2, angle, diff, unwrap, sin, cos, \
std, concatenate, tile, dot, ndarray, transpose, conjugate, ones, \
inf, cumsum, fix, sqrt

from numpy.linalg import inv, norm

from scipy.interpolate import interp1d
from scipy.signal import lfilter
from scipy.io.wavfile import read

from misc import arrayByIndex, mytranspose, end, transpose1dArray, normalize, \
isContainer, isEmpty, erbs2hz, hz2erbs, primes, apply, singlelize, ellipFilter, \
medfilt

from copy import deepcopy

from tqdm import tqdm

from warnings import filterwarnings

from SWIPE import swipep

from scipy.linalg import LinAlgError



def eaQHManalysis(speechFile: str, gender: str = 'other', step: int = 15,
                  maxAdpt: int = 10, pitchPeriods: int = 3, analysisWindow: int = 32, fullWaveform: bool = True,
                  fullBand: bool = True, eaQHM: bool = True, fc: int = 0, partials: int = 0,
                  printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs adaptive Quasi-Harmonic Analysis of Speech
    using the extended adaptive Quasi-Harmonic Model and decomposes 
    speech into AM-FM components according to that model.

    Parameters
    ----------
    speechFile : str
        The location of the mono .wav file to be analysed.
    gender : str, optional
        The gender of the speaker. Can also be 'child'. The default is 'other'.
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
    printPrompts : bool, optional
        Determines if prompts of this process will be printed. The default is True.
    loadingScreen : bool, optional
        Determines if a tqdm loading screen will be displayed in the console. The default is True.

    Returns
    -------
    Determ : (No_ti) array_like
        The Deterministic part of the signal. An array containing elements of Deterministic-class type.
    Stoch : (No_ti) array_like
        The Stochastic part of the signal. An array containing elements of Stochastic-class type. If fullWaveform == True, an empty array is returned.
    Var : Various
        Various important data.
    SRER : (maxAdpt+1) array_like
        An array containing all the adaptation numbers of the signal.
    aSNR : (maxAdpt, No_ti) array_like
        An array containing each SNR (Signal to Noise Ratio) of each time instant per adaptation.

    '''
    startTime = time()
    filterwarnings("ignore")
    min_interp_size = 4
    
    fs, s = read(speechFile)
    s = transpose1dArray(s/normalize)
    length = len(s)
    
    if printPrompts:
        if eaQHM:
            print('\nPerforming extended adaptive Quasi Harmonic Model analysis in file: {}\n'.format(speechFile))
        else:
            print('\nPerforming adaptive Quasi Harmonic Model analysis in file: {}\n'.format(speechFile))
    
    if fc > 0:
        if printPrompts:
            print('High pass filtering at', fc, 'Hz is applied.')
        s = transpose(ellipFilter(transpose(s), fs, fc))
        
    s2 = deepcopy(s)
        
    if gender == 'male':
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
    
    try:
        f0s = swipep(transpose(s2)[0], fs, speechFile, [f0min, f0max], 0.001, -inf)
    except LinAlgError:
        if printPrompts:
            print("Initial SWIPEP failed. Using swipep2.\n")
        opt_swipep = {
            "dt": 0.001,
            "plim": [f0min, f0max],
            "dlog2p": 1/96,
            "dERBs": 0.1,
            "woverlap": 0.5,
            "sTHR": -inf
        }
        f0s = swipep2(transpose(s2)[0], fs, speechFile, opt_swipep, printPrompts, loadingScreen)
    opt_pitch_f0min = f0min        

    f0sin = getLinear(f0s, arange(0, len(s2)-1, round(fs*5/1000))/fs)
       
    if fullBand:
        Fmax = int(fs/2-200)
        
        if partials > 0:
            Kmax = partials
        else:
            Kmax = int(round(Fmax/min(f0sin[:,1])) + 10)
    else:
        Fmax = int(fs/2-2000)
        Kmax = int(round(Fmax/min(f0sin[:,1])) + 10) 
    
    
    analysisWindowSamples = analysisWindow*step
    
    P, p_step = voicedUnvoicedFrames(s, fs, gender)

    if not fullWaveform:
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
        for i, p in enumerate(P):
            if p.ti > analysisWindowSamples/2 and p.isSpeech and (not p.isVoiced) and p.ti < length - analysisWindowSamples/2:
                P[i].isVoiced = True
            if p.ti > analysisWindowSamples/2 and (not p.isSpeech) and (not p.isVoiced) and p.ti < length - analysisWindowSamples/2:
                P[i].isSpeech = True
                P[i].isVoiced = True
        deterministic_part = s

    ti = arange(1,length,step)
    No_ti = len(ti)

    Determ = [Deterministic() for _ in range(No_ti)]
    Var = Various(s=s, fs=fs, fullBand=fullBand, Kmax=Kmax, Fmax=Fmax, filename=speechFile, fullWaveform=fullWaveform)
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
            
            if tith > analysisWindowSamples and tith < length-analysisWindowSamples:
                if P[pf[i]-1].isVoiced and P[pf[i]].isVoiced: 
                    if m == 0:
                        lamda = p[i] - pf[i]
                        
                        f0 = (1-lamda)*f0sin[pf[i]-1][1] + lamda*f0sin[pf[i]][1]
                        
                        K = int(min(Kmax, int(Fmax/f0)))
                        
                        fk = arange(-K,K+1)*f0
                        
                        N[i] = max(120, round((pitchPeriods/2)*(fs/f0))) 
                        
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
                        
                        if not eaQHM:
                            ak_tmp, bk_tmp, aSNR[m][i] = aqhmLS_complexamps(s[n + tith], fm_tmp, win, fs)
                        else:
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
                    Determ[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=True)
                else:
                   Determ[i] = Deterministic(ti=tith-1, isSpeech=True, isVoiced=False)
            else:
                Determ[i] = Deterministic(ti=tith-1, isSpeech=False, isVoiced=False)
            
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
            Var.qh = s_hat
        a0_fin = a0_hat
        am_fin = am_hat
        fm_fin = fm_hat
        pm_fin = pm_hat
        
    for i, d in enumerate(Determ):
        if d.isVoiced:
            ti = d.ti
            idx = argwhere(am_fin[ti])
            Determ[i].a0 = a0_fin[ti]
            Determ[i].ak = arrayByIndex(idx, am_fin[ti, idx])
            Determ[i].fk = arrayByIndex(idx, fm_fin[ti, idx])
            Determ[i].pk = arrayByIndex(idx, pm_fin[ti, idx])
            
    Stoch = []
    if not fullWaveform:
        #----NOT TESTED----
        from scikits.talkbox import lpc
        from numpy.fft import fft
        from scipy.signal import filtfilt
        from structs import Stochastic
        from misc import maxfilt, peak_picking
        
        Stoch = [Stochastic() for _ in range(No_ti)]
        
        s_noi = s - s_hatT
        
        s_noi = ellipFilter(s, fs, 1500)
        
        M = 25
        s_env = maxfilt(abs(s), 9)
    
        s_noi_env = filtfilt(ones(M)/M, 1, abs(s_noi));    
        
        Var.noi_env = s_noi_env
        
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
                    
                    Stoch[i] = Stochastic(ti=tith-1, isSpeech=True, ap = ap, env_ak = abs(am_s), env_fk = fm_s, env_pk = angle(am_s))
                else:
                    Stoch[i] = Stochastic(ti=tith-1, isSpeech=False)
            else:
                Stoch[i] = Stochastic(ti=tith-1, isSpeech=False)
            if loadingScreen:
                stochloop.update(1)
    
        if loadingScreen:
            stochloop.close()

    if printPrompts:        
        print('Signal adapted to {} dB SRER'.format(round(max(SRER), 6)))
        print('Total Time: {}\n\n'.format(strftime("%H:%M:%S", gmtime(time() - startTime))))
    
    return Determ, Stoch, Var, SRER, aSNR

def eaQHMsynthesis(Determ, Stoch, Var, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs speech synthesis using the extended adaptive Quasi-Harmonic
    Model parameters from eaQHManalysis and resynthesizes speech from 
    its AM-FM components according to the eaQHM model.

    Parameters
    ----------
    Determ : (No_ti) array_like
        The Deterministic part of the signal. An array containing elements of Deterministic-class type.
    Stoch : (No_ti) array_like
        The Stochastic part of the signal. An array containing elements of Stochastic-class type. If fullWaveform == True, an empty array is returned.
    Var : Various
        Various important data.
    printPrompts : bool, optional
        Determines if prompts of this process will be printed. The default is True.
    loadingScreen : bool, optional
        Determines if a tqdm loading screen will be displayed in the console. The default is True.

    Returns
    -------
    s : array_like
        The reconstructed speech signal.
    '''
    startTime = time()
    min_interp_size = 4
    
    Kmax = Var.Kmax
    len1 = Determ[len(Determ)-1].ti + 500
    fs= Var.fs
    
    length = len1
    
    ti = zeros(len(Determ), int)
    a0 = zeros(len1, float)
    am = zeros((length, Kmax), float)
    fm = zeros((length, Kmax), float)
    pm = zeros((length, Kmax), float)
    pm_tmp = zeros((length, Kmax), float)
    am_tmp = zeros((length, Kmax), float)
    
    for i, d in enumerate(Determ):
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
        print('Performing extended adaptive Quasi Harmonic Model synthesis in file: {}'.format(Var.filename))

    
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
        
    if not Var.fullBand:
        #----NOT TESTED----
        from numpy import hanning
        from random import random
        N = Stoch[1].ti - Stoch[0].ti
        n = arange(-N-1, N)
        win = hanning(2*N+1)
        
        for st, i in enumerate(Stoch):
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
    
    return s

def iqhmLS_complexamps(s, fk, win, fs: int, iterates: int = 0):
    '''
    Computes iteratively the parameters of first order complex polynomial
    model using Least Squares. 

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    fk : array_like
        The estimated frequencies.
    win : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.
    iterates : int, optional
        The number of iterations. The default is 0.

    Returns
    -------
    ak : array_like
        Amplitude of harmonics.
    bk : array_like
        Slope of harmonics.
    df : array_like
        Frequency mismatch.
    SNR : float
        Signal-to-noise ratio.

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

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    fm : array_like
        The estimated instantaneous frequencies.
    win : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.

    Returns
    -------
    ak : array_like
        Amplitude of harmonics.
    bk : array_like
        Slope of harmonics.
    SNR : float
        Signal-to-noise ratio.

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
    model using Least Squares, a AM and a FM model for the frequency 

    Parameters
    ----------
    s : array_like
        The part of the signal to be computed.
    am : array_like
        The estimated instantaneous amplitudes.
    fm : array_like
        The estimated instantaneous frequencies.
    win : array_like
        The window of the signal to be computed.
    fs : int
        The sampling frequency.

    Returns
    -------
    ak : array_like
        Amplitude of harmonics.
    bk : array_like
        Slope of harmonics.
    SNR : float
        Signal-to-noise ratio.

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

    Parameters
    ----------
    fm_hat : array_like
        The instantaneous frequencies.
    pm_hat : array_like
        The instantaneous phases.
    idx : array_like
        The indices to be interpolated.

    Returns
    -------
    pm_final : array_like
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
    P : array_like
        An array of structures containing the time instants, if they are voiced and if they are speech.
    p_step : int
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
    
def swipep2(x, fs: int, speechFile: str, opt: dict, printPrompts: bool = True, loadingScreen: bool = True):
    '''
    Performs a pitch estimation of the signal for every time instant.

    Parameters
    ----------
    x : array_like
        The signal to be estimated.
    fs : int
        The sampling frequency.
    speechFile : str
        The file name.
    opt : dict
        A dictionary containing all necessary parameters for the function to run. It must contain the following parameters:
        ---- dt: float - Estimation is performed every dt seconds
        ---- plim: array[2] - An array containing 2 numbers, the first being the minimum frequency
            where the pitch is searched and the second being the maximum frequency
        ---- dlog2p: float - The units the signal is distributed in samples on a base-2 logarithm scale. 
        ---- dERBs: float - The step size of ERBs
        ---- woverlap: float - The overlap of the Hanning window
        ---- sTHR: int - The threshold of the pitch strength.
        
    printPrompts : bool, optional
        Determines if prompts of this process will be printed. The default is True.
    loadingScreen : bool, optional
        Determines if a tqdm loading screen will be displayed in the console. The default is True.


    Returns
    -------
    (3,...) array_like
        An array containing the time instants, the estimation for each instant and the strength of each estimation.

    '''
    from numpy import power, empty, polyfit, polyval, nan
    from misc import myHann, arrayMax, mySpecgram
    
    t = arange(0, len(x)/fs+opt['dt'], opt['dt'])
    
    log2pc = arange(log2(opt['plim'][0]), log2(opt['plim'][1]), opt['dlog2p'])
                
    pc = power(2, log2pc)
    
    S = zeros((len(pc), len(t)))
    
    logWs = apply(log2(divide(8*fs, opt['plim'])), round)
    
    ws = power(2, arange(logWs[0], logWs[1]-1, -1))
    
    pO = divide(8*fs,ws)
    
    d = 1 + log2pc - log2(pO[0])
    
    fERBs = erbs2hz(arange(hz2erbs(min(pc)/4), hz2erbs(fs/2), opt['dERBs']))
    
    if loadingScreen:
        wsloop = tqdm(total=len(ws), position=0, leave=True)
                      
    for i in range(len(ws)):
        if loadingScreen:
            wsloop.set_description("SWIPEP".format(i))
        
        dn = int(max(1, round(8*(1-opt['woverlap'])*fs/pO[i])))
        
        xzp = concatenate((zeros((int(ws[i]/2))), x, zeros((dn + int(ws[i]/2)))))
        
        w = myHann(ws[i])
        o = int(max(0, round(ws[i] - dn)))
        
        X, f, ti = mySpecgram(xzp, int(ws[i]), fs, w, o)

        ip = i+1
        if len(ws) == 1:
            j = transpose1dArray(apply(pc, int))
            k = []
        elif i == len(ws)-1:
            j = argwhere(d - ip > -1)
            k = argwhere(d[j] - ip < 0)
        elif i == 0:
            j = argwhere(d - ip < 1)
            k = argwhere(d[j] - ip > 0)
        else:
            j = argwhere(abs(d - ip) < 1)
            k = arange(0, len(j))
        
        fERBs = fERBs[singlelize(argwhere(fERBs > pc[j[0]]/4)[0]):]
        
        L = sqrt(arrayMax(0, interp1d(f, abs(X), kind=3, fill_value=0, axis=0)(fERBs)))
       
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j]) 
        
        if len(Si[0]) > 1:
            Si = interp1d(ti, Si, 'linear', fill_value=nan)(t)
        else:
            Si = empty((len(Si), len(t)), dtype=object)
            
        if isContainer(k[0]):
            k = k[:, 0]
        
        lamda = d[ j[k] ] - (i+1)
        
        mu = ones( len(j) )
        mu[k] = 1 - abs( transpose(lamda) )
        
        S[transpose(j)[0],:] += multiply(transpose(tile(mu,(len(Si[1]), 1))), Si)
        
        if loadingScreen:
            wsloop.update(1)
           
    if loadingScreen:
        wsloop.close()
       
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
    
    return concatenate((transpose1dArray(t), p, s), axis=1)

def pitchStrengthAllCandidates(f, L, pc):
    from copy import deepcopy
    
    S = zeros((len(pc), len(L[1])))
    k = zeros(len(pc)+1, dtype=int)
    for j in range(len(k)-1):
        k[j+1] = k[j] + argwhere( f[k[j]:] > pc[j]/4)[0]
        

    k = k[1:]
    
    N = sqrt(flipud(cumsum(flipud(multiply(L,L)), axis=0)))
    
    for j in range(len(pc)):
        n = deepcopy(N[k[j], :])
        n[n==0] = inf
        NL = L[k[j]:,:] / tile( n, (len(L)-k[j], 1))
        S[j,:] = pitchStrengthOneCandidate(f[k[j]:], NL, pc[j])
    
    return S

def pitchStrengthOneCandidate(f, NL, pc):
    n = int(singlelize(fix( end(f)/pc - 0.75 )))
    if n==0:
        return None
    
    k = zeros(len(f))
    q = f / pc
    
    pr = concatenate(([1], primes(n)))
    for i in pr:
        a = abs( q - i )
        p = argwhere(a < .25)
        k[p] = cos(2*pi*q[p])
        v = argwhere((0.25 < a) & (a < 0.75))
        k[v] = k[v] + cos( 2*pi * q[v] ) / 2
    k = k * apply(1/f, sqrt)
    k = k / norm( k[k>0] )
    
    return dot(transpose(k), NL)

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

