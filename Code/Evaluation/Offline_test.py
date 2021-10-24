


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

budget_list = [
    300,400,500,600,700,
    300,400,500,600,700,
    300,400,500,600,700,
    300,400,500,600,700,
    300,400,500,600,700,
    300,400,500,600,700,
]
name = [
    '300RCT_Chi_final','400RCT_Chi_final', '500RCT_Chi_final','600RCT_Chi_final', '700RCT_Chi_final',
    '300RCT_CTS_final','400RCT_CTS_final','500RCT_CTS_final','600RCT_CTS_final','700RCT_CTS_final',
    '300RCT_ED_final','400RCT_ED_final','500RCT_ED_final','600RCT_ED_final','700RCT_ED_final',
    '300RCT_CF_DT_final','400RCT_CF_DT_final','500RCT_CF_DT_final','600RCT_CF_DT_final','700RCT_CF_DT_final',
    '300RCT_CT_ST_final','400RCT_CT_ST_final','500RCT_CT_ST_final','600RCT_CT_ST_final','700RCT_CT_ST_final',
    '300RCT_LBCF_final','400RCT_LBCF_final','500RCT_LBCF_final','600RCT_LBCF_final','700RCT_LBCF_final'
]
file_name = [
    '300RCT_Chi_final','400RCT_Chi_final', '500RCT_Chi_final','600RCT_Chi_final', '700RCT_Chi_final',
    '300RCT_CTS_final','400RCT_CTS_final','500RCT_CTS_final','600RCT_CTS_final','700RCT_CTS_final',
    '300RCT_ED_final','400RCT_ED_final','500RCT_ED_final','600RCT_ED_final','700RCT_ED_final',
    '300RCT_CF_DT_final','400RCT_CF_DT_final','500RCT_CF_DT_final','600RCT_CF_DT_final','700RCT_CF_DT_final',
    '300RCT_CT_ST_final','400RCT_CT_ST_final','500RCT_CT_ST_final','600RCT_CT_ST_final','700RCT_CT_ST_final',
    '300RCT_LBCF_final','400RCT_LBCF_final','500RCT_LBCF_final','600RCT_LBCF_final','700RCT_LBCF_final'
]

file_name[0] = pd.read_csv('../Model/Chi_ED_CTS/300RCT_Chi_final.csv')
file_name[1] = pd.read_csv('../Model/Chi_ED_CTS/400RCT_Chi_final.csv')
file_name[2] = pd.read_csv('../Model/Chi_ED_CTS/500RCT_Chi_final.csv')
file_name[3] = pd.read_csv('../Model/Chi_ED_CTS/600RCT_Chi_final.csv')
file_name[4] = pd.read_csv('../Model/Chi_ED_CTS/700RCT_Chi_final.csv')

file_name[5] = pd.read_csv('../Model/Chi_ED_CTS/300RCT_CTS_final.csv')
file_name[6] = pd.read_csv('../Model/Chi_ED_CTS/400RCT_CTS_final.csv')
file_name[7] = pd.read_csv('../Model/Chi_ED_CTS/500RCT_CTS_final.csv')
file_name[8] = pd.read_csv('../Model/Chi_ED_CTS/600RCT_CTS_final.csv')
file_name[9] = pd.read_csv('../Model/Chi_ED_CTS/700RCT_CTS_final.csv')

file_name[10] = pd.read_csv('../Model/Chi_ED_CTS/300RCT_ED_final.csv')
file_name[11] = pd.read_csv('../Model/Chi_ED_CTS/400RCT_ED_final.csv')
file_name[12] = pd.read_csv('../Model/Chi_ED_CTS/500RCT_ED_final.csv')
file_name[13] = pd.read_csv('../Model/Chi_ED_CTS/600RCT_ED_final.csv')
file_name[14] = pd.read_csv('../Model/Chi_ED_CTS/700RCT_ED_final.csv')

file_name[15] = pd.read_csv('../Model/CF_DT/300RCT_CF_DT_final.csv')
file_name[16] = pd.read_csv('../Model/CF_DT/400RCT_CF_DT_final.csv')
file_name[17] = pd.read_csv('../Model/CF_DT/500RCT_CF_DT_final.csv')
file_name[18] = pd.read_csv('../Model/CF_DT/600RCT_CF_DT_final.csv')
file_name[19] = pd.read_csv('../Model/CF_DT/700RCT_CF_DT_final.csv')

file_name[20] = pd.read_csv('../Model/CT_ST/300RCT_CT_ST_final.csv')
file_name[21] = pd.read_csv('../Model/CT_ST/400RCT_CT_ST_final.csv')
file_name[22] = pd.read_csv('../Model/CT_ST/500RCT_CT_ST_final.csv')
file_name[23] = pd.read_csv('../Model/CT_ST/600RCT_CT_ST_final.csv')
file_name[24] = pd.read_csv('../Model/CT_ST/700RCT_CT_ST_final.csv')

file_name[25] = pd.read_csv('../Model/LBCF/300RCT_LBCF_final.csv')
file_name[26] = pd.read_csv('../Model/LBCF/400RCT_LBCF_final.csv')
file_name[27] = pd.read_csv('../Model/LBCF/500RCT_LBCF_final.csv')
file_name[28] = pd.read_csv('../Model/LBCF/600RCT_LBCF_final.csv')
file_name[29] = pd.read_csv('../Model/LBCF/700RCT_LBCF_final.csv')



PMG = dict()
name_list=['Chi','CTS','ED','CF_DT','CT_ST','LBCF']
for i in range(30):
    df_overlap_summary = file_name[i][['label','tmp_label']].groupby('tmp_label') \
           .agg({'tmp_label':'size', 'label':'mean'}) \
           .rename(columns={'tmp_label':'count','label':'mean'}) \
           .reset_index()
    df_stg_summary = file_name[i].groupby('final_coin')\
                    .count().reset_index().rename(columns={'treatment':'num_stg_treatment'})[['final_coin','num_stg_treatment']]
    df_summary = df_overlap_summary.merge(df_stg_summary,how = 'inner',left_on = 'tmp_label', right_on = 'final_coin')
    total_gain = sum(df_summary['mean'] * df_summary['num_stg_treatment'])
    total_count = df_summary.num_stg_treatment.sum()
    avg_gain = total_gain/total_count
    base = file_name[i][file_name[i].treatment == 0.1].label.mean()
    gain = (avg_gain - base)/base
    m = int(i/5)
    PMG[name[i]] = [name_list[m],budget_list[i],gain]
    

df_final = pd.DataFrame.from_dict(PMG,orient='index').reset_index()
df_final.columns = ['fullname','Methods','Budget','Percentage Mean Gain']


sns.lineplot(data=df_final, x="Budget", y="Percentage Mean Gain", hue="Methods")
# plt.legend(loc='right')
# plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],['0', '1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%'])
# # plt.show()
plt.savefig('../../Images/Percentage Mean Gain')
# plt.close()






