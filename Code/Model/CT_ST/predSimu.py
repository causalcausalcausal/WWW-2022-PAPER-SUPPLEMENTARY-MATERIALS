import pandas as pd
from math import inf

def find_rule(rule, side):
    if side == 'down':
        ind = 0
    elif side == 'up':
        ind = 1
    else:
        raise ValueError
    lim = rule.split(';')[ind].strip()
    if lim == '-Inf':
        return -inf
    elif lim == 'Inf':
        return inf
    else:
        return float(lim)

def find_grp(h1, h2, h3, h4, rule_df):
    tmp = rule_df[(rule_df.H1_down <= h1)
                  & (rule_df.H1_up >= h1)
                  & (rule_df.H2_down <= h2)
                  & (rule_df.H2_up >= h2)
                  & (rule_df.H3_down <= h3)
                  & (rule_df.H3_up >= h3)
                  & (rule_df.H4_down <= h4)
                  & (rule_df.H4_up >= h4)]
    if tmp.shape[0] != 1:
        return -1
    return tmp['grp_ind'].values[0]


def findRuleGrpInd(x, rule_df):
    tmp = rule_df[(rule_df.H4_hi > x) & (rule_df.H4_lo <= x)]
    if tmp.shape[0] != 1:
        raise ValueError
    return tmp.rule_grp_ind.values[0]

def calGrpEffect(df_grp):
    return pd.Series(index=['value_gain'], data=[df_grp[df_grp.A == 1].Value.mean() - df_grp[df_grp.A == 0].Value.mean()])

data_dir = './data_prep/'
synthetic_data_dir = '../../../Data/Synthetic_data/'

for uw in [5, 10, 15, 20]:
    # uw = 5
    fn_test = '%dWeight_SimulationDataAllTest.csv' % uw
    df_test_all = pd.read_csv(synthetic_data_dir + fn_test)
    df_test_all = df_test_all[df_test_all.A == 0]
    for treatment in ['A1', 'A2', 'A3']:
        # treatment = 'A1'
        print('processing uw = %d treatment = %s' % (uw, treatment))
        rule_fn = '%d%sValuedecisionTable.csv' % (uw, treatment)
        fn1 = '%s_%dWeight_SimulationDataTraining.csv' % (treatment, uw)
        fn2 = '%s_%dWeight_SimulationDataTest.csv' % (treatment, uw)
        df_rule = pd.read_csv(data_dir + rule_fn)
        df_data_train = pd.read_csv(data_dir + fn1)
        df_data_test = pd.read_csv(data_dir + fn2)
        df_data_test = df_data_test[df_data_test.A == 0]
        df_rule['H4_hi'] = list(map(lambda x: float(x.split(';')[1].strip('[').strip(')').strip('(').strip()), df_rule.H4))
        df_rule['H4_lo'] = list(map(lambda x: float(x.split(';')[0].strip('[').strip(')').strip('(').strip()), df_rule.H4))
        df_rule['rule_grp_ind'] = list(range(1,df_rule.shape[0]+1))
        df_data_train['rule_grp_ind'] = list(map(lambda x: findRuleGrpInd(x, df_rule), df_data_train.H4))
        df_data_test['rule_grp_ind'] = list(map(lambda x: findRuleGrpInd(x, df_rule), df_data_test.H4))
        df_grp_effect = df_data_train.groupby('rule_grp_ind').apply(calGrpEffect).reset_index()
        df_treat_effect = pd.merge(df_data_test, df_grp_effect, on='rule_grp_ind')
        df_treat_effect = df_treat_effect[['ID', 'value_gain']].rename(index=str, columns={'value_gain':treatment + '_value_gain'})
        df_test_all = df_test_all.merge(df_treat_effect, on='ID')
        print(df_test_all.shape)
    df_test_all.to_csv(data_dir + '%dSimulationDataTestResult.csv' % uw, index=False)