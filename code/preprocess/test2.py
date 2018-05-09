#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:03:06 2018

@author: Mint
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:11:47 2018

@author: dohoonkim
"""

import pandas as pd
import numpy as np
import os
import glob
import dask.dataframe as dd

#allFiles = glob.glob(os.path.join(delta_path,"*.csv"))
#np_array_list = []
#
#for file_ in allFiles:
#    df = pd.read_csv(file_,index_col=None, header=0)
#    np_array_list.append(df.as_matrix())
#
#big_frame = pd.DataFrame(np_array_list)
#
#big_frame.columns = ["min","max","avg"]
#delta index 0  = min, index1 = max, index2 = mean

#concatenate min,max,mean and sleep stage labels
stage = pd.read_csv('/Users/Mint/Desktop/labelins2.csv',header=None)

deltamin = pd.read_csv('/Users/Mint/Desktop/Spyder/deltamin.csv',header=None)
deltamax = pd.read_csv('/Users/Mint/Desktop/Spyder/deltamax.csv',header=None)
deltamean = pd.read_csv('/Users/Mint/Desktop/Spyder/deltamean.csv',header=None)

correctlen = len(deltamin.columns)
stagere = stage.loc[:,0:correctlen-1]

delta = pd.concat([deltamin,deltamax,deltamean,stagere],ignore_index=True)
delta = delta.transpose()
#grab columns corresponding to the sleep stage labels

Delta0 = delta.loc[delta[3] == 0]
MeanDelta0 = delta.loc[delta[3] == 0].mean()
StdDelta0 = delta.loc[delta[3] == 0].std()
Delta1 = delta.loc[delta[3] == 1]
MeanDelta1 = delta.loc[delta[3] == 1].mean()
StdDelta1 = delta.loc[delta[3] == 1].std()
Delta2 = delta.loc[delta[3] == 2]
MeanDelta2 = delta.loc[delta[3] == 2].mean()
StdDelta2 = delta.loc[delta[3] == 2].std()
Delta3 = delta.loc[delta[3] == 3]
MeanDelta3 = delta.loc[delta[3] == 3].mean()
StdDelta3 = delta.loc[delta[3] == 3].std()
Delta5 = delta.loc[delta[3] == 5]
MeanDelta5 = delta.loc[delta[3] == 5].mean()
StdDelta5 = delta.loc[delta[3] == 5].std()
#WakevalsDelta = delta.loc[delta[3] == 0 or 1 or 2 or 3 or 5]
#MericaDelta = delta.loc[delta[3] == 1 or 2 or 3]
#SpiegelhalderDelta = delta.loc[delta[3] == 2]
NumRow0 = delta.loc[delta[3] == 0]
NumRow1 = delta.loc[delta[3] == 1]
NumRow2 = delta.loc[delta[3] == 2]
NumRow3 = delta.loc[delta[3] == 3]
NumRow4 = delta.loc[delta[3] == 4]
NumRowR = delta.loc[delta[3] == 5]

#0.5-14 Hz (low)
lowmin = pd.read_csv('/Users/Mint/Desktop/Spyder/lowmin.csv',header=None)
lowmax = pd.read_csv('/Users/Mint/Desktop/Spyder/lowmax.csv',header=None)
lowmean = pd.read_csv('/Users/Mint/Desktop/Spyder/lowmean.csv',header=None)

low = pd.concat([lowmin,lowmax,lowmean,stagere],ignore_index=True)
low = low.transpose()
#MericaLow = low.loc[low[3 == 0 or 1 or 2 or 3 or 5]]
Low0 = low.loc[low[3] == 0]
Low1 = low.loc[low[3] == 1]
Low2 = low.loc[low[3] == 2]
Low3 = low.loc[low[3] == 3]
Low5 = low.loc[low[3] == 5]

Low0 = low.loc[low[3] == 0]
LowMean0 = low.loc[low[3] == 0].mean()
LowStd0 = low.loc[low[3] == 0].std()
Low1 = low.loc[low[3] == 1]
LowMean1 = low.loc[low[3] == 1].mean()
LowStd1 = low.loc[low[3] == 1].std()
Low2 = low.loc[low[3] == 2]
LowMean2 = low.loc[low[3] == 2].mean()
LowStd2 = low.loc[low[3] == 2].std()
Low3 = low.loc[low[3] == 3]
LowMean3 = low.loc[low[3] == 3].mean()
LowStd3 = low.loc[low[3] == 3].std()
Low5 = low.loc[low[3] == 5]
LowMean5 = low.loc[low[3] == 5].mean()
LowStd5 = low.loc[low[3] == 5].std()

#alpha
alphamin = pd.read_csv('/Users/Mint/Desktop/Spyder/alphamin.csv',header=None)
alphamax = pd.read_csv('/Users/Mint/Desktop/Spyder/alphamax.csv',header=None)
alphamean = pd.read_csv('/Users/Mint/Desktop/Spyder/alphamean.csv',header=None)

alpha = pd.concat([alphamin,alphamax,alphamean,stagere],ignore_index=True)
alpha = alpha.transpose()

Alpha2 = alpha.loc[low[3] == 2]
AlphaMean2 = alpha.loc[low[3] == 2].mean()
AlphaStd2 = alpha.loc[low[3] == 2].std()

#sigma
sigmamin = pd.read_csv('/Users/Mint/Desktop/Spyder/sigmamin.csv',header=None)
sigmamax = pd.read_csv('/Users/Mint/Desktop/Spyder/sigmamax.csv',header=None)
sigmamean = pd.read_csv('/Users/Mint/Desktop/Spyder/sigmamean.csv',header=None)

sigma = pd.concat([sigmamin,sigmamax,sigmamean,stagere],ignore_index=True)
sigma = sigma.transpose()

SigmaMean0 = sigma.loc[low[3] == 0].mean()
SigmaStd0 = sigma.loc[low[3] == 0].std()

#beta
betamin = pd.read_csv('/Users/Mint/Desktop/Spyder/betamin.csv',header=None)
betamax = pd.read_csv('/Users/Mint/Desktop/Spyder/betamax.csv',header=None)
betamean = pd.read_csv('/Users/Mint/Desktop/Spyder/betamean.csv',header=None)

beta = pd.concat([betamin,betamax,betamean,stagere],ignore_index=True)
beta = beta.transpose()

Beta0 = beta.loc[beta[3] == 0]
BetaMean0 = beta.loc[low[3] == 0].mean()
BetaStd0 = beta.loc[low[3] == 0].std()
Beta1 = beta.loc[beta[3] == 1]
BetaMean1 = beta.loc[low[3] == 1].mean()
BetaStd1 = beta.loc[low[3] == 1].std()
Beta2 = beta.loc[beta[3] == 2]
BetaMean2 = beta.loc[low[3] == 2].mean()
BetaStd2 = beta.loc[low[3] == 2].std()
Beta3 = beta.loc[beta[3] == 3]
BetaMean3 = beta.loc[low[3] == 3].mean()
BetaStd3 = beta.loc[low[3] == 3].std()
#MericaPerlisBeta = beta.loc[beta[3] == 1 or 2 or 3]
#FreedmanBeta = beta.loc[beta[3] == 0]j
#FreedmanBeta1 = beta.loc[beta[3] == 1]
#FreedmanBeta.append(FreedmanBeta1)
#
#SpiegelhalderBeta = beta.loc[beta[3] == 2]
