import dask.dataframe as dd
import sys
import os
import random
import pandas as pd
import numpy as np


for i in [5,10,15,20]:
    predict_1=pd.read_csv("./output/MBCF_uplift_"+str(i)+"uw_1.csv",header=None)
    predict_1.columns=['predict']
    predict_2=pd.read_csv("./output/MBCF_uplift_"+str(i)+"uw_2.csv",header=None)
    predict_2.columns=['predict']
    predict_3=pd.read_csv("./output/MBCF_uplift_"+str(i)+"uw_3.csv",header=None)
    predict_3.columns=['predict']
    test=pd.read_csv("./data/test_data_"+str(i)+"uw.csv",sep=' ',header=None)
    test.columns=['H1','H2','H3','H4',"exp_group","value"]
    test['predict_1']=predict_1['predict']
    test['predict_2']=predict_2['predict']
    test['predict_3']=predict_3['predict']
    test=test[['exp_group','predict_1','predict_2','predict_3']]
    test.to_csv("./output/MBCF_uplift_"+str(i)+"uw_merged.csv",index=False)


predict_1=pd.read_csv("./output/MBCF_uplift_RCT1.csv",header=None)
predict_1.columns=['predict']
predict_2=pd.read_csv("./output/MBCF_uplift_RCT2.csv",header=None)
predict_2.columns=['predict']
predict_3=pd.read_csv("./output/MBCF_uplift_RCT3.csv",header=None)
predict_3.columns=['predict']
predict_4=pd.read_csv("./output/MBCF_uplift_RCT4.csv",header=None)
predict_4.columns=['predict']
predict_5=pd.read_csv("./output/MBCF_uplift_RCT5.csv",header=None)
predict_5.columns=['predict']
predict_6=pd.read_csv("./output/MBCF_uplift_RCT6.csv",header=None)
predict_6.columns=['predict']
predict_7=pd.read_csv("./output/MBCF_uplift_RCT7.csv",header=None)
predict_7.columns=['predict']
test=pd.read_csv("../../../Data/RCT_data/test_data_MBCF.csv",sep=' ',header=None)
test['predict_1']=predict_1['predict']
test['predict_2']=predict_2['predict']
test['predict_3']=predict_3['predict']
test['predict_7']=predict_7['predict']
test['predict_4']=predict_4['predict']
test['predict_5']=predict_5['predict']
test['predict_6']=predict_6['predict']
test=test[[15,'predict_1','predict_2','predict_3','predict_4','predict_5','predict_6','predict_7']]
test.columns=['exp_group','predict_1','predict_2','predict_3','predict_4','predict_5','predict_6','predict_7']
test.to_csv("./output/MBCF_uplift_RCT_merged.csv",index=False)
# predict_multi=pd.read_csv("WWW2021_MAPG_multi_predict_10w_1022.csv",header=None)
# predict_multi.columns=['predict_1','predict_2','predict_3','predict_4','predict_5','predict_6','predict_7']
# test=pd.read_csv("WWWW_MAPG_test_10w.csv")
# test['predict_1']=predict_multi['predict_1']
# test['predict_2']=predict_multi['predict_2']
# test['predict_3']=predict_multi['predict_3']
# test['predict_4']=predict_multi['predict_4']
# test['predict_5']=predict_multi['predict_5']
# test['predict_6']=predict_multi['predict_6']
# test['predict_7']=predict_multi['predict_7']
# # test['predict_4']=predict_4['predict']
# # test.head()
# test=test[['exp_group','predict_1','predict_2','predict_3','predict_4','predict_5','predict_6','predict_7']]
# test.head()
# test.to_csv("WWW2021_MAPG_multi_predict_all_10w_1022.csv",index=False)

