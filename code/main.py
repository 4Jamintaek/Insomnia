#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 01:44:36 2017

@author: J & M

Main script to run the signal process
Change inputs to fit your data

Signal preprocess package work flow:
    1. Process annotation and header files
    2. Process the record to contain the channels of interest
    3. Filter the record using notch filter and low pass fileter
    4. Remove artifact by ICA

Feature extraction package work flow:
    1. Run Time frequency analysis
    2. 
"""

from preprocess import io as in_out
from preprocess import process_data
from sklearn.decomposition import PCA
from scipy.fftpack import fft, ifft
from scipy import signal as SIG
import pandas as pd
import numpy as np
import time as clock
import matplotlib.pyplot as plt

#Measure time 
t, elapsed = [], []
t.append(clock.time())

"""
#Use GUI to pick data
signal = in_out.open_data()
"""

# Just for a debugging purpose
signal = process_data.raw_signal('/users/Mint/Desktop/Spyder/Data/n2.edf',
                                            '/users/Mint/Desktop/Spyder/Data/n2.txt',
                                            '/users/Mint/Desktop/Spyder/Data/Header_file.txt')


#%% Signal Preprocessing parameters
signal.process_annot_n_header()
eeg_ch = ['Fp2-F4','F4-C4','C4-P4','P4-O2','C4-A1']
emg_ch, ecg_ch, eog_ch = 'EMG1-EMG2','ECG1-ECG2',['ROC-LOC','LOC-ROC']
american = False
h_freq = 60 # Just a testing value. Can try different number under Nyquist Frequency
buffer_time = '01:00:00' #Wake time before sleep onset at the beginin and end of the sleep

visualize = True
r_thres = 0.8


#%% Run signal preprocess
signal.process_record(eeg_ch, emg_ch, ecg_ch, eog_ch,american,h_freq,buffer_time)
signal.artifact_removal_ICA(visualize,r_thres)


#Record signals


#Measure preprocess time
elapsed.append(clock.time() - t[-1])
t.append(clock.time())

#%% Feature extraction parameters

#Define the corrected signals
NewData = signal.corrected_data

#Fast Fourier Transform with window size 3 seconds and window overlap 50%


#%% Run feature extraction

def buffer(x, n, p, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Args
    ----
    x:   signal array
    n:   number of data segments
    p:   number of values to overlap
    opt: initial condition options. default sets the first `p` values
         to zero, while 'nodelay' begins filling the buffer immediately.
    '''
    import numpy

    if p >= n:
        raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

    # Calculate number of columns of buffer array
    cols = int(numpy.ceil(len(x) // int(n-p)))

    # Check for opt parameters
    if opt == 'nodelay':
        # Need extra column to handle additional values left
        cols += 1
    elif opt != None:
        raise SystemError('Only `None` (default initial condition) and '
                          '`nodelay` (skip initial condition) have been '
                          'implemented')

    # Create empty buffer array
    b = numpy.zeros((n, cols))
    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(int(cols)):
        # Set first column to n values from x, move to next iteration
        if i == 0 and opt == 'nodelay':
            b[0:n,i] = x[0:n]
            continue
        # set first values of row to last p values
        elif i != 0 and p != 0:
            b[:p, i] = b[-p:, i-1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0

        # Get stop index positions for x
        k = j + n - p

        # Get stop index position for b, matching number sliced from x
        n_end = p+len(x[j:k])

        # Assign values to buffer array from x
        b[p:n_end,i] = x[j:k]

        # Update start index location for next iteration of x
        j = k

    return b

NewSig = signal.corrected_data
NewEEG = signal.corrected_data[0:5,]

import scipy.io as sio

NewDataTruncated=signal.corrected_data[:,:]
sio.savemat('/users/Mint/Desktop/arrdata3.mat', mdict={'NewDataTruncated': NewDataTruncated})
#Code for Pseudocode 1:

#NewSig = signal.corrected_data
#maximum = len(NewSig[0,:])

#firstSig = NewSig[0,:]
#secondSig = NewSig[1,:]
#thirdSig = NewSig[2,:]
#fourthSig = NewSig[3,:]
#fifthSig = NewSig[4,:]
#
#firstList = [firstSig[x:x+7680] for x in range(0, maximum, 7680)]
#secondList = [secondSig[x:x+7680] for x in range(0, maximum, 7680)]
#thirdList = [thirdSig[x:x+7680] for x in range(0, maximum, 7680)]
#fourthList = [fourthSig[x:x+7680] for x in range(0, maximum, 7680)]
#fifthList = [fifthSig[x:x+7680] for x in range(0, maximum, 7680)]


#"""NFFT = [512,256,128]"""

# Number of Sample Points 

#N = len(NewSig[0,:])

# Sample spacing

#T = 1/256
#x = np.linspace(0.0, N*T, N)
#NewEEGfft=fft(NewEEG)


#plt.plot(NewEEGfft)
#plt.ylabel('EEG signal')
#plt.show()

# Get real amplitudes of FFT (only in postive frequencies)
#fft_vals = np.absolute(np.fft.rfft(NewEEG))

# Get frequencies for amplitudes in Hz
#fft_freq = np.fft.rfftfreq(len(NewEEG), 1.0/256)

## Define EEG bands
#
#eeg_bands = {'Delta': (0.5, 3),
#             'Theta': (4, 8),
#             'Alpha': (8.5, 12),
#             'Beta': (16.5, 30),
#             'Gamma': (30.5, 60),
#             'Sigma': (12.5-16),
#             'Decrease': (0.5, 14.75),
#             'Increase': (14.75, 60)}
#
## Take the mean of the fft amplitude for each EEG band
#eeg_band_fft = dict()
#for band in eeg_bands:
#    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
#                       (fft_freq <= eeg_bands[band][1]))[0]
#    eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
#
#    
## Plot the data
#
#df = pd.DataFrame(columns=['band', 'val'])
#df['band'] = eeg_bands.keys()
#df['val'] = [eeg_band_fft[band] for band in eeg_bands]
#ax = df.plot.bar(x='band', y='val', legend=False)
#ax.set_xlabel("EEG band")
#ax.set_ylabel("Mean band Amplitude")




#Output512 = np.zeros((5,7027200))

#"""
#have to modify to incorporate cases for different window sizes
#Output256 = np.array(5,7027200)
#Output128 = np.array(5,7027200)
#"""
#"""for i = 0:2"""
#
#x = np.zeros((5,7027200))
#X = np.zeros((7027200,1))
#mean512 = np.zeros((5,NFFT))
#for j in range(0,5):
#    win = SIG.hamming(NFFT)    
#    x[j,:] = NewSig[j,:]
#    """X[:,j] = x[j,:].transpose()"""
#    pval = int(NFFT//2)
#    bx = buffer(x[j,:],NFFT,pval)
#    bx = bx[:,1:len(bx[0])-1]
#    a = np.diag(win)
#    bx = np.dot((np.transpose(bx)),a)
#    meanFFT = abs(fft(bx,NFFT) / sum(win))
#    mean = meanFFT.mean(0)
#    mean512[j,:] = np.transpose(mean)
#    
##%%
#
#import pandas as pd
#from sklearn.decomposition import PCA    
#    
#feat_cols = eeg_ch
#mean512 = np.transpose(mean512)   
#d = {'Fp2-F4': mean512[:,0], 'F4-C4': mean512[:,1], 'C4-P4': mean512[:,2], 'P4-O2': mean512[:,3],'C4-A1': mean512[:,4]}
#df = pd.DataFrame(data = d)
#
#pca = PCA(n_components=3)
#pca_result = pca.fit_transform(df.values)
#
#df['pca-one'] = pca_result[:,0]
#df['pca-two'] = pca_result[:,1] 
#df['pca-three'] = pca_result[:,2]
#
#df1 = df[df.columns[5:8]]
#
#print (pca.explained_variance_ratio_)
#rndperm = np.random.permutation(df.shape[0])
#    
#
#"""bx = np.dot((np.transpose(bx)),win)"""
#"""length = len(bx[0])"""
#"""bx = np.transpose(bx).dot(np.diag(win))"""
#"""meanFFT = abs(fft(bx,NFFT) / sum(win))"""
#"""meanFFT = np.mean(meanFFT,2)"""
#"""Output512[j,:] = meanFFT"""