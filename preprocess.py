"""
Functions relating to data retrevial and processing.

Created on Sat Jun 11 00:34:03 2022

@author: dunn
"""
import numpy as np
import pandas as pd
import pybaselines.spline as py
import scipy.signal as ss
from scipy.sparse import csc_matrix, eye, diags #smooth.whitaker
from scipy.sparse.linalg import spsolve #smooth.whitaker

def z_score(intensity):
    '''
    Basic Z-Score Function

    Parameters
    ----------
    intensity

    Returns
    -------
    zscores

    '''        
    mean=np.mean(intensity)
    std=np.std(intensity)
    zscores1=(intensity-mean)/std
    zscores=np.array(abs(zscores1))
    return(zscores)
def mod_zscore(intensity):
    '''
    Parameters
    ----------
    intensity

    Returns
    -------
    mod_zscores

    '''
    median_int=np.median(intensity)
    mad_int=np.median([np.abs(intensity-median_int)])
    mod_z_scores1=0.6745*(intensity-median_int)/mad_int
    mod_z_scores=np.array(abs(mod_z_scores1))
    return mod_z_scores

def WhitakerHayes_zscore(intensity, threshold):
    '''
    Whitaker-Hayes Function uses Intensity Modified Z-Scores

    Parameters
    ----------
    intensity : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    intensity_modified_zscores : TYPE
        DESCRIPTION.

    '''
    dist=0
    delta_intensity=[]
    for i in np.arange(len(intensity)-1):
        dist=intensity[i+1]-intensity[i]
        delta_intensity.append(dist)
    delta_int=np.array(delta_intensity)
    
    #Run the delta_int through MAD Z-Score Function
    intensity_modified_zscores=np.array(np.abs(mod_zscore(delta_int)))
    return intensity_modified_zscores

def detect_spike(z, threshold):
    '''
    #Add threshold parameter and use in zap function, right now this is redundant (6/2022)
    
    It detects spikes, or sudden, rapid changes in a spectral intensity.
    #potential edit: automatically generate spike graph
    
    Parameters
    ----------
    z : int/float
        Input z-score

    Returns
    -------
    spikes : Bool
        TRUE = spikes, FALSE = non-spikes.
        Potential edit: automatically generate spike graphs

    '''
    spikes=abs(abs(np.array(z))>threshold) #True assigned to spikes, False to non-spikes
    return (spikes)

def fix(y, m=2, threshold=5):
    '''
    Replace spike intensity values with the average values that are not spikes in the selected range

    Parameters
    ----------
    y : Input intensity
        DESCRIPTION.
    m : int, selected range, optional
        Selction of points around the detected spike. The default is 2.
    threshold : TYPE, optional
        Binarization threshold. Increase value will increase spike detection sensitivity.
        I think.
        The default is 5.
        
    Returns
    -------
    y_out : float/array
        Average values that are around spikes in the selected range (~m)
    '''
    spikes=abs(np.array(mod_zscore(np.diff(y))))>threshold
    y_out=y.copy() #Prevents overeyeride of input y
    for i in np.arange(len(spikes)):
      if spikes[i] !=0: #If there is a spike detected in position i
        w=np.arange(i-m, i+1+m) #Select 2m+1 points around the spike
        w2=w[spikes[w]==0] #From the interval, choose the ones which are not spikes
        if not w2.any(): #Empty array
            y_out[i]=np.mean(y[w]) #Average the values that are not spikes in the selected range        
        if w2.any(): #Normal array
            y_out[i]=np.mean(y[w2]) #Average the values that are not spikes in the selected range
    return y_out

def process(y):
    #2) Fix Spectra: remove spikes from cosmic rays
    try:
        y1=fix(y)
    except IndexError: #Bias IndexError for when spike is detected at edges of data
        y1=y
        pass

    #3) Smooth Intensity
    yZS=ss.savgol_filter(y1, 9, 3)

    #4) Apply baseline
    b_out=py.pspline_asls(yZS)[0]
    Y=yZS-b_out   
    
    #5) Processing Figure
    # fig=figure.preprocess(x, y0, Y, b_out, fullname)

    #6) Returns    
    return Y