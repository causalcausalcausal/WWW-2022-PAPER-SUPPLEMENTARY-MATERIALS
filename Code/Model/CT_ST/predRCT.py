
RCT_data_dir = '../../../Data/RCT_data/'


import pandas as pd

def findRuleGrpInd(x, rule_df):
    tmp = rule_df[(rule_df.H4_hi > x) & (rule_df.H4_lo <= x)]
    if tmp.shape[0] != 1:
        raise ValueError
    return tmp.rule_grp_ind.values[0]

def calGrpEffect(df_grp):
    return pd.Series(index=['value_gain'], data=[df_grp[df_grp.A == 1].num_of_days.mean() - df_grp[df_grp.A == 0].num_of_days.mean()])

treatmentList = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
data_dir = './data_prep/'
test_data_fn = 'WWWW_MAPG_test.csv'
df_data_test = pd.read_csv(RCT_data_dir + test_data_fn)

for treatment in treatmentList:
    rule_fn = '%s_RCTnum_of_daysLeafPath.csv' % treatment
    train_data_fn = '%sRCTDataTraining.csv' % treatment

    df_data_train = pd.read_csv(data_dir + train_data_fn)

    path_list = []
    with open(data_dir + rule_fn, 'r') as f:
        for row in f:
            if row[0] != 'c':
                continue
            path_list.append(row[2:-2])

    leaf_path_list = []
    for i in range(len(path_list)):
        unique = True
        for j in range(len(path_list)):
            if i == j:
                continue
            if path_list[i] in path_list[j]:
                unique = False
                print('del',i, j)
                break
        if unique:
            leaf_path_list.append(path_list[i])


    df_data_train_indexed_list = []
    i = 0
    for path in leaf_path_list:
        df_tmp = df_data_train.copy()
        df_tmp['rule_grp_ind'] = i
        for rule in path.split(','):
            rule = rule.strip('"')
            if rule == "root":
                continue
            elif '<' in rule:
                if '<=' in rule:
                    print(rule)
                    print(path)
                    raise ValueError
                col = rule.split('<')[0].strip().strip('"')
                val = float(rule.split('<')[1].strip().strip('"'))
                df_tmp = df_tmp[df_tmp[col] < val]
            elif '>' in rule:
                if '>=' not in rule:
                    print(rule)
                    print(path)
                    raise ValueError
                col = rule.split('>=')[0].strip().strip('"')
                val = float(rule.split('>=')[1].strip().strip('"'))
                df_tmp = df_tmp[df_tmp[col] >= val]
            else:
                print(rule)
                print(path)
                raise ValueError
        df_data_train_indexed_list.append(df_tmp.copy())
        i += 1

    df_data_test_indexed_list = []
    i = 0
    for path in leaf_path_list:
        df_tmp = df_data_test.copy()
        df_tmp['rule_grp_ind'] = i
        for rule in path.split(','):
            rule = rule.strip('"')
            if rule == "root":
                continue
            elif '<' in rule:
                if '<=' in rule:
                    print(rule)
                    print(path)
                    raise ValueError
                col = rule.split('<')[0].strip().strip('"')
                val = float(rule.split('<')[1].strip().strip('"'))
                df_tmp = df_tmp[df_tmp[col] < val]
            elif '>' in rule:
                if '>=' not in rule:
                    print(rule)
                    print(path)
                    raise ValueError
                col = rule.split('>=')[0].strip().strip('"')
                val = float(rule.split('>=')[1].strip().strip('"'))
                df_tmp = df_tmp[df_tmp[col] >= val]
            else:
                print(rule)
                print(path)
                raise ValueError
        df_data_test_indexed_list.append(df_tmp.copy())
        i += 1
    df_data_train_indexed = pd.concat(df_data_train_indexed_list)
    df_data_test_indexed = pd.concat(df_data_test_indexed_list)
    df_data_test_indexed = df_data_test_indexed[['ID', 'rule_grp_ind']]
    df_grp_effect = df_data_train_indexed.groupby('rule_grp_ind').apply(calGrpEffect).reset_index()
    df_data_test_effect = df_data_test_indexed.merge(df_grp_effect, on='rule_grp_ind')[['ID', 'value_gain']].\
        rename(index=str, columns={'ID':'ID', 'value_gain':'%s_value_gain'%treatment})
    df_data_test = df_data_test.merge(df_data_test_effect, on='ID')

df_data_test.to_csv(data_dir + 'realRTreePartitionedTest.csv',index=False)


