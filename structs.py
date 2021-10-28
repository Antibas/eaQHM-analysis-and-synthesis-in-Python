# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:00:39 2021

@author: Panagiotis Antivasis
"""
class Stochastic:
    def __init__(self, ti = [], isSpeech = False, ap = [], env_ak = [], env_fk = [], env_pk = []):
        self.ti = ti
        self.isSpeech = isSpeech
        self.ap = ap
        self.env_ak = env_ak
        self.env_fk = env_fk
        self.env_pk = env_pk
        
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()

class Deterministic:
    def __init__(self, ti = [], isSpeech = False, isVoiced = False, a0 = [], ak = [], fk = [], \
                 pk = []):
        self.ti = ti
        self.isSpeech = isSpeech
        self.isVoiced = isVoiced
        self.a0 = a0
        self.ak = ak
        self.fk = fk
        self.pk = pk

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()

class Frame:
    def __init__(self, ti, isSpeech, isVoiced):
        self.ti = ti
        self.isSpeech = isSpeech
        self.isVoiced = isVoiced
    
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()

class Various:
    def __init__(self, s = [], fs = 0, qh = [], env = [], noi_env = [], \
                 fullBand = False, Kmax = 0, Fmax = 0, filename = '', fullWaveform = False):
        self.s = s
        self.fs = fs
        self.env = env
        self.noi_env = noi_env
        self.fullBand = fullBand
        self.Kmax = Kmax
        self.Fmax = Fmax
        self.filename = filename
        self.fullWaveform = fullWaveform
    
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()