# CT.ST README

This model utilized R code from https://github.com/tuye0305/prophet for model prediction, and python for data processing.
Please check libraries in main.R for R environment setup, and pandas for python environment.

Homepath in main.R should be set before running. 
The steps for simulation data are as follow:

    python dataPrepSimu.py
    Rscript main.R
    python predSimu.py
    python CT_ST_budget_allocation.py

For RCT data, plz change the dataType in main.R file then run as follow:
    
    python dataPrepRCT.py
    Rscript main.R
    python predRCT.py
    python CT_ST_budget_allocation-RCT.py

    
