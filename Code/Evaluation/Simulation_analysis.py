import pandas as pd
import numpy as np
import seaborn as sns


uncertaintyW = [
    '5','10','15','20'
    ,'5','10','15','20'
    ,'5','10','15','20'
    ,'5','10','15','20'
    ,'5','10','15','20'
    ,'5','10','15','20'
]
name = [
    '5Chi','10Chi','15Chi','20Chi'
    ,'5ED','10ED','15ED','20ED'
    ,'5CTS','10CTS','15CTS','20CTS'
    ,'5CF_DT','10CF_DT','15CF_DT','20CF_DT'
    ,'5CT_ST','10CT_ST','15CT_ST','20CT_ST'
    ,'5LBCF','10LBCF','15LBCF','20LBCF'
]
file_name = [
    '5Chi_final','10Chi_final','15Chi_final','20Chi_final'
    ,'5ED_final','10ED_final','15ED_final','20ED_final'
    ,'5CTS_final','10CTS_final','15CTS_final','20CTS_final'
    ,'5CF_DT_final','10CF_DT_final','15CF_DT_final','20CF_DT_final'
    ,'5CT_ST_final','10CT_ST_final','15CT_ST_final','20CT_ST_final'
    ,'5LBCF_final','10LBCF_final','15LBCF_final','20LBCF_final'
]

file_name[0] = pd.read_csv('../Model/Chi_ED_CTS/5Chi_final')
file_name[1] = pd.read_csv('../Model/Chi_ED_CTS/10Chi_final')
file_name[2] = pd.read_csv('../Model/Chi_ED_CTS/15Chi_final')
file_name[3] = pd.read_csv('../Model/Chi_ED_CTS/20Chi_final')
file_name[4] = pd.read_csv('../Model/Chi_ED_CTS/5ED_final')
file_name[5] = pd.read_csv('../Model/Chi_ED_CTS/10ED_final')
file_name[6] = pd.read_csv('../Model/Chi_ED_CTS/15ED_final')
file_name[7] = pd.read_csv('../Model/Chi_ED_CTS/20ED_final')
file_name[8] = pd.read_csv('../Model/Chi_ED_CTS/5CTS_final')
file_name[9] = pd.read_csv('../Model/Chi_ED_CTS/10CTS_final')
file_name[10] = pd.read_csv('../Model/Chi_ED_CTS/15CTS_final')
file_name[11] = pd.read_csv('../Model/Chi_ED_CTS/20CTS_final')
file_name[12] = pd.read_csv('../Model/CF_DT/5CF_DT_final')
file_name[13] = pd.read_csv('../Model/CF_DT/10CF_DT_final')
file_name[14] = pd.read_csv('../Model/CF_DT/15CF_DT_final')
file_name[15] = pd.read_csv('../Model/CF_DT/20CF_DT_final')
file_name[16] = pd.read_csv('../Model/CT_ST/5CT_ST_final')
file_name[17] = pd.read_csv('../Model/CT_ST/10CT_ST_final')
file_name[18] = pd.read_csv('../Model/CT_ST/15CT_ST_final')
file_name[19] = pd.read_csv('../Model/CT_ST/20CT_ST_final')
file_name[20] = pd.read_csv('../Model/LBCF/5LBCF_final')
file_name[21] = pd.read_csv('../Model/LBCF/10LBCF_final')
file_name[22] = pd.read_csv('../Model/LBCF/15LBCF_final')
file_name[23] = pd.read_csv('../Model/LBCF/20LBCF_final')

ITE = dict()
name_list=['Chi','ED','CTS','CF_DT','CT_ST','LBCF']
for i in range(24):
    file_name[i]['ITE'] = file_name[i]['final_treatment_value'] - file_name[i]['value0']
    m = int(i/4)
    ITE[name[i]] = [uncertaintyW[i],name_list[m],file_name[i]['ITE'].sum()/file_name[i]['value0'].sum()]
    
df_final = pd.DataFrame.from_dict(ITE,orient='index').reset_index()
df_final.columns = ['fullname','uncertaintyW','name','ITE']

fig = sns.factorplot(x='uncertaintyW', y='ITE', hue='name', data=df_final, kind='bar')
fig.savefig('../../Images/ITE')





    


