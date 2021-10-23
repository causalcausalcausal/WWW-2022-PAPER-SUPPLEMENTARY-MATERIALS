import pandas as pd
import numpy as np

uncertaintyWList = [5,10,15,20]

for w in uncertaintyWList:
    test = pd.read_csv("../../../Data/Synthetic_data/" + str(w) + "Weight_SimulationDataAllTest.csv" )
    
    test[['H1', 'H2', 'H3', 'H4']]=test[['H1', 'H2', 'H3', 'H4']].round(5).round(1).astype(str)
    test['ID'] = test['ID'].round(0).astype(str)
    
    test_pivot = test.pivot(index=['ID','H1','H2','H3','H4'],columns="A",values=['Value','Cost']).reset_index()
    test_pivot.columns=test_pivot.columns.droplevel()
    test_pivot.columns=['ID', 'H1', 'H2', 'H3', 'H4','value0' ,'value1','value2','value3','cost0' ,'cost1','cost2','cost3']
    
    boolean = test_pivot.duplicated(subset=['ID'])
    if boolean.any():
        print('Duplicate_data')
    
    for model in ['Chi','ED','CTS']:
        
        modelresult = pd.read_csv(str(w)+model+'result',names = ['Dummy','pred1','pred2','pred3'],header=0)
        model_final = test_pivot.copy()
        model_final[['pred1','pred2','pred3']]=modelresult[['pred1','pred2','pred3']]

        # Roi for each treatment
        model_final['t_1_roi'] = model_final['pred1']/model_final['cost1']
        model_final['t_2_roi'] = model_final['pred2']/model_final['cost2']
        model_final['t_3_roi'] = model_final['pred3']/model_final['cost3']

        # For each user i, select the treatment with max ROI
        model_final['best_treatment_roi'] = np.where((model_final[['t_1_roi','t_2_roi','t_3_roi']] < 0).all(axis=1),0,model_final[['t_1_roi','t_2_roi','t_3_roi']].max(axis=1))
        model_final['best_treatment']=np.where((model_final[['t_1_roi','t_2_roi','t_3_roi']] <0).all(axis=1),'control',model_final[['t_1_roi','t_2_roi','t_3_roi']].idxmax(axis=1))

        coin_map = {'control':'value0','t_1_roi': 'value1','t_2_roi':'value2','t_3_roi':'value3'}
        model_final['best_te'] = model_final['best_treatment'].map(coin_map)
        idx, cols = pd.factorize(model_final['best_te'])
        model_final['best_te_value'] = pd.DataFrame(model_final.reindex(cols,axis=1).to_numpy()[np.arange(len(model_final)),idx])
        model_final['best_te_value'] = model_final['best_te_value'].fillna(0)

        coin_map = {'control':'cost0','t_1_roi': 'cost1','t_2_roi':'cost2','t_3_roi':'cost3'}
        model_final['best_te_cost'] = model_final['best_treatment'].map(coin_map)
        idx, cols = pd.factorize(model_final['best_te_cost'])
        model_final['best_te_cost'] = pd.DataFrame(model_final.reindex(cols,axis=1).to_numpy()[np.arange(len(model_final)),idx])
        model_final['best_te_cost'] = model_final['best_te_cost'].fillna(0)

        coin_map = {'control':'control','t_1_roi': 'pred1','t_2_roi':'pred2','t_3_roi':'pred3'}
        model_final['best_te_pred'] = model_final['best_treatment'].map(coin_map)
        idx, cols = pd.factorize(model_final['best_te_pred'])
        model_final['best_te_pred'] = pd.DataFrame(model_final.reindex(cols,axis=1).to_numpy()[np.arange(len(model_final)),idx])
        model_final['best_te_pred'] = model_final['best_te_pred'].fillna(0)

        # Sort all users according to each user's max ROI in a descending order.
        model_final = model_final.sort_values(by=['best_treatment_roi'],ascending=False)

        budget = 1500000

        model_final['cum_expense'] = (model_final['best_te_cost']).cumsum()

        model_final['extra_expense'] = budget
        model_final['cut_off_point'] = model_final['cum_expense'] <= model_final['extra_expense']
        model_final['final_treatment_value'] =  np.where(model_final['cut_off_point'],model_final['best_te_value'],model_final['value0'])

        model_final.to_csv(str(w)+model+'_final')
        print(model+ '_upper_bound: ' + str(model_final.best_te_cost.sum()))