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
stage = pd.read_csv('/Users/dohoonkim/Desktop/Research/labelins3.csv',header=None)

deltamin = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/deltamin.csv',header=None)
deltamax = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/deltamax.csv',header=None)
deltamean = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/deltamean.csv',header=None)

correctlen = len(deltamin.columns)
stagere = stage.loc[:,0:correctlen-1]

delta = pd.concat([deltamin,deltamax,deltamean,stagere],ignore_index=True)
delta = delta.transpose()
#grab columns corresponding to the sleep stage labels
Wakevals = delta.loc[delta[3] == 0]

#theta
thetamin = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/thetamin.csv',header=None)
thetamax = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/thetamax.csv',header=None)
thetamean = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/thetamean.csv',header=None)

theta = pd.concat([thetamin,thetamax,thetamean,stagere],ignore_index=True)
theta = theta.transpose()

#alpha
alphamin = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/alphamin.csv',header=None)
alphamax = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/alphamax.csv',header=None)
alphamean = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/alphamean.csv',header=None)

alpha = pd.concat([alphamin,alphamax,alphamean,stagere],ignore_index=True)
alpha = alpha.transpose()

#sigma
sigmamin = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/sigmamin.csv',header=None)
sigmamax = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/sigmamax.csv',header=None)
sigmamean = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/sigmamean.csv',header=None)

sigma = pd.concat([sigmamin,sigmamax,sigmamean,stagere],ignore_index=True)
sigma = sigma.transpose()

#beta
betamin = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/betamin.csv',header=None)
betamax = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/betamax.csv',header=None)
betamean = pd.read_csv('/Users/dohoonkim/Desktop/Research/insomnia/betamean.csv',header=None)

beta = pd.concat([betamin,betamax,betamean,stagere],ignore_index=True)
beta = beta.transpose()