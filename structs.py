# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:00:39 2021

@author: Panagiotis Antivasis
"""
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