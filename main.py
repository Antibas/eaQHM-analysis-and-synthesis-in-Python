# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:16:16 2021

@author: Panagiotis Antivasis
"""

from functions import eaQHMAnalysisAndSynthesis
from numpy import arange, float32
from scipy.io.wavfile import write
from matplotlib.pyplot import plot, show, xlabel, ylabel, title, specgram, colorbar
from scipy.io.wavfile import read
from misc import normalize
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def plotGraphs(t, t_reconst, signal, signal_reconst, name, fs):
    colorbar(label="Intensity (dB)")
    title("Spectrogram of " + name)
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    show()
    
    specgram(signal_reconst, Fs=fs, vmin=-180, vmax=-40)
    colorbar(label="Intensity (dB)")
    title("Spectrogram of " + name + " reconstructed")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    show()
    
    plot(t, signal)
    title(name)
    xlabel('Time (s)')
    ylabel('Amplitude')
    show()
    
    plot(t_reconst, signal_reconst)
    title(name + ' reconstructed')
    xlabel('Time (s)')
    ylabel('Amplitude')
    show()
    
def main():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = askopenfilename(
        parent=root,
        title="Write the name of the file to be processed", 
        initialdir="/", 
        filetypes=("WAVFILES *.wav",))
    
    if filename == "":
        raise ValueError("No File Selected")
        
    print("File Selected: " + filename)
    gender = input("You may include a gender (male, female, child or other): ")
    
    print()
    
    signal_reconstructed, SRER, DetComponents, time = eaQHMAnalysisAndSynthesis(filename, gender, loadingScreen=False)
    
    fs, signal = read(filename)
    signal = signal/normalize
    dt = 1/fs
    t = arange(0, len(signal)/fs, dt)
    
    t_reconstructed = arange(0, len(signal_reconstructed)/fs, dt)
    plotGraphs(t, t_reconstructed, signal, signal_reconstructed, filename, fs)
    
    write(filename[0:len(filename)-4]+"_reconstructed.wav", fs, float32(signal_reconstructed))
    
if __name__ == "__main__":
    main()