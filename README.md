# LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm

This repo contains all the data, code files and pics mentioned in Supplementary Materials of paper "**LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm**".

## **Reproduction Instructions**

#### ***Section 5.1 Simulation Analysis***

Steps to reproduce the results:

1. Add your work directory to "homePath" in generateSimulationData.R under Code/Data_generation folder.
2. Run generateSimulationData.R.
3. Follow the instruction under each model's folder for training and prediction; run budget_allocation.py for each model.
4. Run Simulation_analysis.py under Code/Evaluation folder.


#### ***Section 5.2 Offline Test***

Steps to reproduce the results:


1. xx
2. xx
3. xx


## **Implementation Details**


1. The proposed CATE estimation method **UDCF** (**U**nified **D**iscriminative **C**ausal **F**orest) is implemented by directly modifying the C++ source code of **GRF** (**G**eneralized **R**andom **F**orests) with a new splitting criterion. Therefore, UDCF requires a C++ running environment as GRF. Detailed info for environment configuration plz refers to GRF [Github Pages](https://github.com/grf-labs/grf).
2. The proposed MCKP distributed algorithm **DGB** (**D**ual **G**radient **B**isection) is written in Python and run on Pytorch.
3. The baseline methods: uplift random forests on **ED** (**E**uclidean **D**istance), **Chi** (**Chi**-Square) and **CTS** (**C**ontextual **T**reatment **S**election) are all directly imported from CausalML package [Github Pages](https://github.com/uber/causalml).
4. The baseline methods: **CT.ST** (**C**ausal **T**ree with **S**tochastic Op**t**imization) and **CF.DT** (**C**ausal **F**orest with **D**eterministic Op**t**imization) proposed by Tu et al. [1] are implemented by following the algorithm listed in the paper [1]. R package is required.


## **Reference**
[1]. Ye Tu, Kinjal Basu, Cyrus DiCiccio, Romil Bansal, Preetam Nandy, Padmini Jaikumar, and Shaunak Chatterjee. 2021. Personalized Treatment Selection using Causal Heterogeneity. In Proceedings of the Web Conference 2021. 1574–1585.
