from causalml.inference.tree import UpliftRandomForestClassifier
import pandas as pd
import numpy as np

uncertaintyWList = [5,10,15,20]

for w in uncertaintyWList:
    train = pd.read_csv(str(w)+'Weight_SimulationDataAllTraining.csv')
    test = pd.read_csv(str(w)+'Weight_SimulationDataAllTest.csv')
    train.A = train.A.astype(str)
    features = ['H1', 'H2', 'H3', 'H4']
    
    
    Chi_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "Chi", 
                                            max_depth = 5,min_samples_leaf=100, 
                                            min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
    Chi_uplift_model.fit(X=train[features].values, treatment=train['A'].values, y=train['Value'].values)
    Chi_pred = Chi_uplift_model.predict(test[features].values)
    Chi_test_result = pd.DataFrame(Chi_pred,columns=Chi_uplift_model.classes_)
    Chi_test_result.to_csv(str(w)+'Chiresult')
    
    ED_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "ED", 
                                            max_depth = 5,min_samples_leaf=100, 
                                            min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
    ED_uplift_model.fit(X=train[features].values, treatment=train['A'].values, y=train['Value'].values)
    ED_pred = ED_uplift_model.predict(test[features].values)
    ED_test_result = pd.DataFrame(ED_pred,columns=ED_uplift_model.classes_)
    ED_test_result.to_csv(str(w)+'EDresult')
    
    CTS_uplift_model = UpliftRandomForestClassifier(n_estimators=300, evaluationFunction = "CTS", 
                                            max_depth = 5,min_samples_leaf=100, 
                                            min_samples_treatment=50,n_reg=100,control_name='0', n_jobs=1,normalization=True)
    CTS_uplift_model.fit(X=train[features].values, treatment=train['A'].values, y=train['Value'].values)
    CTS_pred = CTS_uplift_model.predict(test[features].values)
    CTS_test_result = pd.DataFrame(CTS_pred,columns=CTS_uplift_model.classes_)
    CTS_test_result.to_csv(str(w)+'CTSresult')
    
    
    