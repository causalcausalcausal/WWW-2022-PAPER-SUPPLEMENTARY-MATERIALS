import dask.dataframe as dd
import sys
import os
import random
import pandas as pd
import numpy as np

def getT(x):
    if(x.A==0):
        return 0
    return 1
def getT1(x):
    if(x.A==1):
        return 1
    return 0
def getT2(x):
    if(x.A==2):
        return 1
    return 0
def getT3(x):
    if(x.A==3):
        return 1
    return 0
def getT4(x):
    if(x.A==4):
        return 1
    return 0
def getT0(x):
    if(x.A==0):
        return 1
    return 0


##train
for i in [5,10,15,20]:
    df=pd.read_csv("../../../Data/Synthetic_data/"+str(i)+"Weight_SimulationDataAllTraining.csv")
    df=df[['H1','H2','H3','H4','A','Value']]
    df['T1']=df.apply(getT1,axis=1)
    df['T2']=df.apply(getT2,axis=1)
    df['T3']=df.apply(getT3,axis=1)
    #df_1=df[(df['A']==0) | (df['A']==1)]
    #df_2=df[(df['A']==0) | (df['A']==2)]
    #df_3=df[(df['A']==0) | (df['A']==3)]
    #df_1['T']=df_1.apply(getT,axis=1)
    #df_2['T']=df_2.apply(getT,axis=1)
    #df_3['T']=df_3.apply(getT,axis=1)
    #df_1=df_1[['H1','H2','H3','H4','T','Value']]
    #df_2=df_2[['H1','H2','H3','H4','T','Value']]
    #df_3=df_3[['H1','H2','H3','H4','T','Value']]
    #df_1.to_csv("./MBCF_train_5uw_1.csv",header=None,sep=' ',index=False)
    #df_2.to_csv("./MBCF_train_5uw_2.csv",header=None,sep=' ',index=False)
    #df_3.to_csv("./MBCF_train_5uw_3.csv",header=None,sep=' ',index=False)
    df[['H1','H2','H3','H4','Value','T1','T2','T3']].to_csv("./data/train_data_"+str(i)+"uw.csv",header=None,index=False,sep=' ')

    ##test
    test=pd.read_csv("../../../Data/Synthetic_data/"+str(i)+"Weight_SimulationDataAllTest.csv")
    test=test[['H1','H2','H3','H4','A','Value']]
    test.to_csv("./data/test_data_"+str(i)+"uw.csv",header=None,sep=' ',index=False)
    # test.to_csv("./test_data_5uw.csv",header=None,sep=' ',index=False)

