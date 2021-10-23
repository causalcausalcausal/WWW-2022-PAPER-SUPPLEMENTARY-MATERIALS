rm(list = ls())      # Clear all variables
graphics.off()    # Close graphics windows
gc()
# Loading required packages
library(simcausal)
library(causalTree)
library(grf)
library(randomForest)
library(rosqp)
library(MASS)
library(Matrix)
library(data.table)
library(jsonlite)

library(ggplot2)
library(foreach)
library(doParallel)
library(rattle)
library(lpSolve)
library(osqp)
library(olpsR)

checkResultData <- function(data){
  (is.data.frame(data) && nrow(data)!=0) || (!is.data.frame(data) && !is.na(data))
}


homePath <- "/Users/yunz/Code/www_lbcf/OfflineEvaluation/Code/Model/CT.ST/" #TODO: Please specify the home directory for the project

dataType = 'Simu'
# dataType = 'RCT'
if (dataType == 'Simu'){
  treatmentList <- c("A1", "A2", "A3")
  outcomeList <- c("Value", "Cost")
  objective.outcome <- "Value"
  uncertaintyWList = c(5, 10, 15, 20)
} else {
  outcomeList <- c("num_of_days")
  treatmentList <- c("A1", "A2", "A3", "A4", "A5", "A6", "A7")
  objective.outcome <- "num_of_days"
  uncertaintyWList = c(2)
}

source(paste0(homePath, "cohortLevelProphetUtils.R"))
source(paste0(homePath, "mergeTreeUtils.R"))

seed <- 12345

set.seed(seed)

minLeafSize<- 100
numTreatment <- length(treatmentList)


numEpoc <- 10

# dataType = 'RCT'

# uncertaintyW = 10
for (uncertaintyW in uncertaintyWList){
  print(paste0('processing uncertaintyW = ', uncertaintyW  ))
  tau <- if (uncertaintyW <= 1) 0.005 else 0.001/(uncertaintyW^2) # hyperparameter for constraints violation level
  alpha <- 0.01 # hyperparameter of the learning rate
  
  causaltreeMergedStochasticPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  causaltreeStochasticPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  causalTreeDeterministicPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  causalForestDeterministicPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  deltaModelPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  fixedParameterPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  heuristicStochasticPolicyResultDF <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
  
  # one epoc
  trainCausalTree(homePath, outcomeList, treatmentList, minLeafSize, uncertaintyW, dataType)
  
  if (dataType == 'Simu') {
    #merging trees
    Y1.ATE <- -Inf
    causaltreeMergedStochasticPolicyResults <- data.frame("Y1" = as.numeric(), "Y2" = as.numeric(), "Y3" = as.numeric())
    
    dt.list <-list()
    cnt <- 1
    for (treatment in treatmentList){
      for (outcome in outcomeList){
        decisionTable <- fread(paste0(homePath,'data_prep/',uncertaintyW, treatment, outcome, 'decisionTable.csv') , header = TRUE)
        dt.list[[cnt]] <- decisionTable
        cnt <- cnt + 1
      }
    }
    
    #impute data
    dt.out.list <- imputeCols(dt.list)
    dt.table.final <- mergeTrees(dt.out.list)
    decisionTable<- dt.table.final
    write.csv(dt.table.final, file = paste(homePath,'data_prep/', uncertaintyW, 'MergedDecisionTable.csv', sep = ""), row.names = FALSE, quote = FALSE)
  }
}
