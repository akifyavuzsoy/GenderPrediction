# -*- coding: utf-8 -*-
"""

Feature Extraction of Audio Signals




Created on Apr 2024

@author: Akif Yavuzsoy
"""

import librosa                                            # Ses Dosyasının Yüklenmesi
import librosa.display
#import IPython.display as ipd                             # Ses Oynatılması
import matplotlib.pyplot as plt


class Features:
    def __init__(self):
        super().__init__()
        
        
 
    def showSignals(self, x, sr):
        """
        Description
        -----------
        Function showing signals..
        
        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)
 
        Returns
        -------
        None.

        """
        
        plt.figure(figsize=(10,2))
        librosa.display.waveshow(x, sr=sr)
        
        
    def spektrogram(self, x, sr):
        """
        Description
        -----------
        epresents the signal strength or loudness of a signal at various frequencies present in a particular waveform..
        
        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)

        Returns
        -------
        Xdb: Amplitude Decibel value

        """
        
        stft = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(stft))
        return Xdb
    
    
    def mfcc(self, x, sr, n_mfcc):
        """
        Description
        -----------
        It is a scale that shows the human ear's perception of changes in sound frequencies..
        It is the expression of the short-time power spectrum of the audio signal on the Mel scale.
        
        Frekanstan Mel ölçeğine dönüşüm formülü;
            M ═ 1125 × ln(1+(f÷700))
            M → Mel ölçeği
            f → Frekans(Hz)

        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)

        Returns
        -------
        mfcc: Mel-Frequency Cepstral Coefficients

        """
        
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    
    def zeroCrossingRate(self, x):
        """
        Description
        -----------
        It is the rate at which a signal passes through the zero line, that is, the rate of sign change..

        Parameters
        ----------
        x : audio time series

        Returns
        -------
        zero_crossing: a signal passes through the zero line
        
        """
        
        zero_crossing = librosa.zero_crossings(x)
        return sum(zero_crossing)
    
    
    def spectralCentroid(self, x):
        """
        Description
        -----------
        Shows where the center of mass is

        Parameters
        ----------
        x : audio time series

        Returns
        -------
        spec_cent: center of mass

        """
        
        spec_cent = librosa.feature.spectral_centroid(y=x)
        return spec_cent
    
    
    def spectralRolloff(self, x, sr):
        """
        The measure of the signal shape. It represents a certain percentage of the total spectral energy..

        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)

        Returns
        -------
        spec_roll: measure of the signal shape

        """
        
        spec_roll = librosa.feature.spectral_rolloff(y=x, sr=sr)
        return spec_roll
    
    
    def chroma(self, x, sr):
        """
        The spectrum is a powerful representation of sound, with 12 parts representing 12 different halftones (chromas) of the musical octave.

        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)

        Returns
        -------
        chroma: a powerful representation of sound

        """
        
        chroma=librosa.feature.chroma_stft(y=x,sr=sr)
        return chroma
    
    
    def spectralBant(self, x, sr):
        """
        Defines half the maximum peak of the audio signal's wavelength.

        Parameters
        ----------
        x : audio time series
        sr : sound frequency(Hz)

        Returns
        -------
        spec_band: maximum peak of the audio signal's wavelength

        """
        
        spec_band=librosa.feature.spectral_bandwidth(y=x,sr=sr)
        return spec_band
    
    
