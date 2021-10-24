import pandas as pd
import numpy as np
from DGB_RCT import DGB

test = pd.read_csv("../../../Data/RCT_data/rct_test.csv" )
test_result = pd.read_csv("data_prep/RCTTreePartitionedTest.csv")

test[['pred1','pred2','pred3','pred4','pred5','pred6','pred7']]=test_result[['A1_value_gain','A2_value_gain', 
                                             'A3_value_gain', 'A4_value_gain', 'A5_value_gain',
                                             'A6_value_gain', 'A7_value_gain']]
test_pivot = test.copy()

cost = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 1.9])
value = test_pivot[['pred1','pred2','pred3','pred4','pred5','pred6','pred7']].to_numpy()

for budget in [300,400,500,600,700]:
    model = DGB(budget,cost,value)
    model.train()
    df_dgb = model.save_to_dataframe(test_pivot)
    indices, values, spend = model.generate_decisions()
    df_dgb['final_coin'] = df_dgb['cost']
    coin_map = {0:0.1,1: 0.4,2:0.6,3:0.8,4:1.1,5:1.4,6:1.7,7:2.0}
    test_pivot['treatment'] = test_pivot['exp_group'].map(coin_map)
    test_pivot['final_coin'] = test_pivot['final_coin'].round(1)
    test_pivot['treatment'] = test_pivot['treatment'].round(1)
    test_pivot['if_same'] = test_pivot['treatment'] == test_pivot['final_coin']
    test_pivot['tmp_label'] =  np.where(test_pivot['if_same'],test_pivot['treatment'],0)
    test_pivot.to_csv(str(budget)+'RCT_CT_ST_final.csv')