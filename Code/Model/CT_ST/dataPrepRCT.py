import pandas as pd
from math import inf
RCT_data_dir = '../../../Data/RCT_data/'
output_data_dir = './data_prep/'
fn1 = 'WWWW_MAPG_%s.csv'
fn2 = 'WWWW_MAPG_test.csv'

for type in ['Training', 'Test']:
    df = pd.read_csv(RCT_data_dir + 'WWWW_MAPG_%s.csv' % type.lower())
    df0 = df[df.exp_group == 0]
    df1 = df[df.exp_group == 1]
    df2 = df[df.exp_group == 2]
    df3 = df[df.exp_group == 3]
    df4 = df[df.exp_group == 4]
    df5 = df[df.exp_group == 5]
    df6 = df[df.exp_group == 6]
    df7 = df[df.exp_group == 7]

    df_1_out = pd.concat([df1, df0])
    df_2_out = pd.concat([df2, df0])
    df_3_out = pd.concat([df3, df0])
    df_4_out = pd.concat([df4, df0])
    df_5_out = pd.concat([df5, df0])
    df_6_out = pd.concat([df6, df0])
    df_7_out = pd.concat([df7, df0])

    df_1_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_1_out['exp_group']))
    df_2_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_2_out['exp_group']))
    df_3_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_3_out['exp_group']))
    df_4_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_4_out['exp_group']))
    df_5_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_5_out['exp_group']))
    df_6_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_6_out['exp_group']))
    df_7_out['A'] = list(map(lambda x: 0 if x == 0 else 1, df_7_out['exp_group']))

    df_1_out.to_csv(output_data_dir + "A1" + "RCTData%s.csv" % type)
    df_2_out.to_csv(output_data_dir + "A2" + "RCTData%s.csv" % type)
    df_3_out.to_csv(output_data_dir + "A3" + "RCTData%s.csv" % type)
    df_4_out.to_csv(output_data_dir + "A4" + "RCTData%s.csv" % type)
    df_5_out.to_csv(output_data_dir + "A5" + "RCTData%s.csv" % type)
    df_6_out.to_csv(output_data_dir + "A6" + "RCTData%s.csv" % type)
    df_7_out.to_csv(output_data_dir + "A7" + "RCTData%s.csv" % type)
