# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:16:16 2021

@author: Panagiotis Antivasis
"""

from functions import eaQHManalysis, eaQHMsynthesis
from numpy import arange, float32
from scipy.io.wavfile import write
from matplotlib.pyplot import subplots, show

def plot(t, t_reconst, signal, signal_reconst, name):
    fig, (ax1, ax2) = subplots(2, sharex=True)
    
    ax1.plot(t, signal)
    ax1.set_title(name)
    ax1.set_ylabel('Amplitude')
    ax2.plot(t_reconst, signal_reconst)
    ax2.set_title(name+' after eaQHM Reconstruction')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    
    show()

def main():
    filename = 'af049orgh_snd_norm_F'#input("Write the name of the file to be processed: ")
    root = '../thesis_files/mat_files/'#'path/to/the/parameter_and_wav_files' #change according to the location of the files
   
    D, S, V, SRER, aSNR = eaQHManalysis(root+filename+".wav", root+filename+"_parameters", loadingScreen=False)
    signal = V.s
    fs = V.fs
    dt = 1/fs
    t = arange(0, len(signal)/fs, dt)
    
    signal_reconstructed, qh, noi = eaQHMsynthesis(D, S, V, loadingScreen=False)
    t_reconstructed = arange(0, len(signal_reconstructed)/fs, dt)
    plot(t, t_reconstructed, signal, signal_reconstructed, V.filename)
    
    write(root+filename+"_reconstructed.wav", fs, float32(signal_reconstructed))
    
if __name__ == "__main__":
    main()