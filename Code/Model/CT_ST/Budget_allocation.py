import pandas as pd
import numpy as np
from DGB import DGB

uncertaintyWList = [5,10,15,20]

for w in uncertaintyWList:
    test = pd.read_csv("../../../Data/Synthetic_data/" + str(w) + "Weight_SimulationDataAllTest.csv" )
    test_result = pd.read_csv("data_prep/"+str(w)+"SimulationDataTestResult.csv")

    test[['H1', 'H2', 'H3', 'H4']]=test[['H1', 'H2', 'H3', 'H4']].round(5).round(1).astype(str)
    test['ID'] = test['ID'].astype(int)

    test_pivot = test.pivot(index=['ID','H1','H2','H3','H4'],columns="A",values=['Value','Cost']).reset_index()
    test_pivot.columns=test_pivot.columns.droplevel()
    test_pivot.columns=['ID', 'H1', 'H2', 'H3', 'H4','value0' ,'value1','value2','value3','cost0' ,'cost1','cost2','cost3']

    boolean = test_pivot.duplicated(subset=['ID'])
    if boolean.any():
        print('Duplicate_data')

    test_result['pred1'] = test_result['A1_value_gain'] 
    test_result['pred2'] = test_result['A2_value_gain'] 
    test_result['pred3'] = test_result['A3_value_gain'] 
    test_pivot_1=test_result[['ID','pred1','pred2','pred3']]
    test_pivot= test_pivot.merge(test_pivot_1,how = 'inner',on='ID')

    budget = 1500000
    cost = test_pivot[['cost1','cost2','cost3']].to_numpy()
    value = test_pivot[['pred1','pred2','pred3']].to_numpy()
    model = DGB(budget,cost,value)
    model.train()

    df_dgb = model.save_to_dataframe(test_pivot)
    indices, values, spend = model.generate_decisions()
    true_value = []
    for i in indices:
        true_value.append(df_dgb[['value1','value2','value3']].to_numpy()[i])
    true_value_v2 = df_dgb['value0'].to_numpy().copy()
    for ind, val in zip(indices, true_value):
        i, j = ind
        true_value_v2[i] = val
    df_dgb['final_treatment_value'] = true_value_v2   
    df_dgb.to_csv(str(w)+"CF_ST_final")