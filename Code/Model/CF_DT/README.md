
# CF.DT README

This model utilized C++ code for training MBCF models and prediction, and python for data processing.

The steps for simulation data are as follow:

    unzip CF_DT_Synthetic.zip
    python data_preprocessing.py
    follow instructions under ./MBCF_Synthetic for training and prediciton
    python data_merging.py
    python CF_DT_budget_allocation.py
    
The steps for RCT data are as follow:

    unzip CF_DT_RCT.zip
    python data_preprocessing.py
    follow instructions under ./MBCF_RCT for training and prediciton
    python preprocessing.py
    python data_merging.py
    python CF_DT_budget_allocation-RCT.py
