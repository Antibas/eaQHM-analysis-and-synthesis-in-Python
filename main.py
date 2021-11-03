# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:16:16 2021

@author: Panagiotis Antivasis
"""

from functions import eaQHMAnalysisAndSynthesis
from numpy import arange, float32
from scipy.io.wavfile import write
from matplotlib.pyplot import subplots, show
from scipy.io.wavfile import read
from misc import normalize

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
    filename = input("Write the name of the file to be processed: ")
    gender = input("You may include a gender (male, female, child or other): ")
    
    print()
    
    signal_reconstructed, SRER = eaQHMAnalysisAndSynthesis(filename, gender=gender, loadingScreen=False)
    
    fs, signal = read(filename)
    signal = signal/normalize
    dt = 1/fs
    t = arange(0, len(signal)/fs, dt)
    
    t_reconstructed = arange(0, len(signal_reconstructed)/fs, dt)
    plot(t, t_reconstructed, signal, signal_reconstructed, filename)
    
    write(filename[0:len(filename)-4]+"_reconstructed.wav", fs, float32(signal_reconstructed))
    
if __name__ == "__main__":
    main()