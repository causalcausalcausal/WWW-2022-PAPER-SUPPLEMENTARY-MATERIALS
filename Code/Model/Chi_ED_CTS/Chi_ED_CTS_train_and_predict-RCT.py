from causalml.inference.tree import UpliftRandomForestClassifier
import pandas as pd
import numpy as np

train = pd.read_csv("../../../Data/RCT_data/rct_training.csv"  )
test = pd.read_csv("../../../Data/RCT_data/rct_test.csv" )
train.exp_group = train.exp_group.astype(str)
features = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10',
   'c11', 'c12', 'c13']


Chi_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "Chi", 
                                        max_depth = 5,min_samples_leaf=100, 
                                        min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
Chi_uplift_model.fit(X=train[features].values, treatment=train['exp_group'].values, y=train['label'].values)
Chi_pred = Chi_uplift_model.predict(test[features].values)
Chi_test_result = pd.DataFrame(Chi_pred,columns=Chi_uplift_model.classes_)
Chi_test_result.to_csv('RCT_Chiresult')

ED_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "ED", 
                                        max_depth = 5,min_samples_leaf=100, 
                                        min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
ED_uplift_model.fit(X=train[features].values, treatment=train['exp_group'].values, y=train['label'].values)
ED_pred = ED_uplift_model.predict(test[features].values)
ED_test_result = pd.DataFrame(ED_pred,columns=ED_uplift_model.classes_)
ED_test_result.to_csv('RCT_EDresult')

CTS_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "CTS", 
                                        max_depth = 5,min_samples_leaf=100, 
                                        min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
CTS_uplift_model.fit(X=train[features].values, treatment=train['exp_group'].values, y=train['label'].values)
CTS_pred = CTS_uplift_model.predict(test[features].values)
CTS_test_result = pd.DataFrame(CTS_pred,columns=CTS_uplift_model.classes_)
CTS_test_result.to_csv('RCT_CTSresult')
    
    
    