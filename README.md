# LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm

This repo contains all the data, code files and pics mentioned in Supplementary Materials of paper "**LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm**".

## **Reproduction Instructions**

**A.**   Steps to reproduce the results of ***Section 5.1 Simulation Analysis***

1. xx
2. xx
3. xx



**B.**   Steps to reproduce the results of ***Section 5.2 Offline Test***


1. xx
2. xx
3. xx


## **Methods Implementation**


1. The proposed CATE estimation method **UDCF** (**U**nified **D**iscriminative **C**ausal **F**orest) is implemented by directly modifying the core C++ source code of **GRF** (**G**eneralized **R**andom **F**orests) with a new splitting criterion. Therefore, UDCF requires a C++ running environment as GRF. Detailed info for environment configuration plz refers to GRF [Github Pages](https://github.com/grf-labs/grf).
2. The proposed MCKP distributed algorithm **DGB** (**D**ual **G**radient **B**isection) is written in Python and run on Pytorch.
3. The baseline methods: uplift random forests on ED (Euclidean Distance), Chi (Chi-Square) and CTS (Contextual Treatment Selection) are all directly imported from CausalML package [Github Pages](https://github.com/uber/causalml).
4. The baseline methods: 
