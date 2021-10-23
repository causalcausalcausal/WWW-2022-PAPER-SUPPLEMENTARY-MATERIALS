trainCausalTree <- function(homePath, outcomeList, treatmentList, minLeafSize, uncertaintyW,dataType){
  
  splitAlpha <- 0.5
  cvHonest <- T
  cvAlpha <- 0.5
  for (treatment in treatmentList){
    print(treatment)
    # Read in input data
    if (dataType == 'Simu') {
      dataNew <- fread(paste0(homePath, 'data_prep/', treatment,'_', uncertaintyW, 'Weight_SimulationDataTraining.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)  
    } else {
      dataNew <- fread(paste0(homePath, 'data_prep/', treatment,'RCTDataTraining.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)  
      dataNew = subset(dataNew, select = c("c0"	,"c1",	"c2",	"c3",	"c4",	"c5",	"c6",	"c7",	"c8",	"c9",	"c10",	"c11",	"c12",	"c13",	"num_of_days",	"A"))
    }
    
    
    names(dataNew)[names(dataNew) == "A"] <- "lixTreatment"
    for (outcome in outcomeList){
      names(dataNew)[names(dataNew) == outcome] <- "label"
      
      # Create a formula for the tree model
      cm <- colnames(dataNew)
      rm.list <- c("ID", "label", "lixTreatment", outcomeList[!outcomeList %in% c(outcome)])
      rm.cm <- which(cm %in% rm.list)
      # Create a simple linear formula
      ff <- as.formula(paste0("label ~ ", paste(cm[- rm.cm], collapse = " + ")))
      
      truePropen = (nrow(subset(dataNew, lixTreatment == 1)) * 1.0) / nrow(dataNew)
      
      # Train the tree
      tree <- causalTree(ff, data = dataNew, treatment = dataNew$lixTreatment, split.alpha = splitAlpha, split.Rule = "CT", cv.option = "TOT", split.Honest = T, cv.Honest = cvHonest, split.Bucket = T, xval = 3, cp = 0, minsize = minLeafSize, propensity = truePropen, maxdepth=5)
      frm = tree$frame
      ordered <- rev(sort(frm$n, index = TRUE)$ix)
      decisionPathDf = NULL
      for (i in ordered){
        pth <- rpart::path.rpart(tree, nodes = as.numeric(i), print.it = FALSE)
        decisionPathDf = rbind(decisionPathDf, pth)
      }
    
      
      
      if (dataType == 'Simu') {
        write.csv(decisionPathDf, file = paste0(homePath, 'data_prep/', treatment,'_',outcome, 'LeafPath.csv'), row.names = FALSE, quote = FALSE)
        # Get the minimal xerror
        opcp <- tree$cptable[, 1][which.min(tree$cptable[, 4])]
        
        # Prune the tree based on the minimal xerror
        opfit <- prune(tree, opcp)
        
        pdf(paste(homePath, 'data_prep/',treatment, outcome, 'causalTree.pdf', sep = ""))
        rpart.plot(tree)
        rpart.plot(opfit)
        dev.off()
        statsDf <- calculateStatsTree(dataNew, opfit, "lixTreatment", "label")
        statsDf.ordered <- statsDf[order(statsDf$delta),]
        
        labels <- labels(opfit)[- 1]
        uniqueVariables <- unique(lapply(labels, function (x) {unlist(strsplit(x, "<|>|="))[1]}))
        decisionTable <- convertToRules(opfit, uniqueVariables, dataNew)
        
        decisionTableEmpty <- data.frame("rule" = c("Rule 0:"), "delta" = c(0), "cover" = c(nrow(dataNew)), "pcover" = c(1))
        decisionTable.ordered <- if(nrow(decisionTable) > 0) decisionTable[order(decisionTable$delta),] else decisionTableEmpty
        
        decisionTable.expand <- cbind(decisionTable.ordered[,!(names(decisionTable.ordered) %in% c("delta"))], statsDf.ordered)
        write.csv(decisionTable.expand, file = paste(homePath, 'data_prep/',uncertaintyW, treatment, outcome, 'decisionTable.csv', sep = ""), row.names = FALSE, quote = FALSE)
        
        write.csv(statsDf.ordered, file = paste(homePath, 'data_prep/',uncertaintyW, treatment, outcome, 'xlntStats.csv', sep = ""), row.names = FALSE, quote = FALSE)
        names(dataNew)[names(dataNew) == "label"] <- outcome
      } else {
        write.csv(decisionPathDf, file = paste0(homePath, 'data_prep/', treatment,'_RCT',outcome, 'LeafPath.csv'), row.names = FALSE, quote = FALSE)
      }
      
    }
  }
}

calculateStatsTree <- function(df, tree, treatmentVar, label) {
  statsDf <- data.frame("delta" = c(), "deltaPercent" = c(), "pValue" = c(), "variance" = c())
  df$leaves <- predict(object = tree, newdata = df, type = 'vector')
  df$leavesFactor <- factor(round(df$leaves, 4))
  cnt <- 0
  for (l in levels(df$leavesFactor)) {
    cnt <- cnt + 1
    treatmentIds <- which(df$leavesFactor == l & df[[treatmentVar]] == 1)
    controlIds <- which(df$leavesFactor == l & df[[treatmentVar]] == 0)
    y1 <- mean(df[[label]][treatmentIds])
    y0 <- mean(df[[label]][controlIds])
    n1 <- as.numeric(nrow(df[treatmentIds,]))
    n0 <- as.numeric(nrow(df[controlIds,]))
    varY1 <- var(df[[label]][treatmentIds]) / n1
    varY0 <- var(df[[label]][controlIds]) / n0
    
    delta <- y1 - y0
    deltaPercent <- (delta * 1.0) / y0
    variance <- varY1 + varY0
    varianceDeltaPercent <- (1 / y0 ^ 2) * varY1 + (y1 ^ 2 / y0 ^ 4) * varY0
    
    tStatsDeltaPercent <- deltaPercent / sqrt(varianceDeltaPercent)
    
    pValueDeltaPercent <- 2 * pnorm(- abs(tStatsDeltaPercent))
    errorMarginDeltaPercent <- 1.96 * sqrt(varianceDeltaPercent)
    statsDf[cnt, "delta"] <- delta
    statsDf[cnt, "deltaPercent"] <- deltaPercent
    statsDf[cnt, "pValueDeltaPercent"] <- pValueDeltaPercent
    statsDf[cnt, "variance"] <- variance
  }
  statsDf
}

calculateStats <- function(df, outcome) {

  treatmentIds <- which(df[["A"]] == 1)
  controlIds <- which(df[["A"]] == 0)
  y1 <- mean(df[[outcome]][treatmentIds])
  y0 <- mean(df[[outcome]][controlIds])
  n1 <- as.numeric(nrow(df[treatmentIds,]))
  n0 <- as.numeric(nrow(df[controlIds,]))
  varY1 <- var(df[[outcome]][treatmentIds]) / n1
  varY0 <- var(df[[outcome]][controlIds]) / n0
    
  delta <- y1 - y0
  deltaPercent <- (delta * 1.0) / y0
  
  varianceDeltaPercent <- (1 / y0 ^ 2) * varY1 + (y1 ^ 2 / y0 ^ 4) * varY0
  variance <- varY1 + varY0
  tStatsDeltaPercent <- deltaPercent / sqrt(varianceDeltaPercent)
  
  pValueDeltaPercent <- 2 * pnorm(- abs(tStatsDeltaPercent))
  errorMarginDeltaPercent <- 1.96 * sqrt(varianceDeltaPercent)
  n <- n0 + n1
  delta <- if (n == 0) 0 else delta
  variance <- if (n == 0) 0 else if (is.na(variance)) var(df[[outcome]])/n else variance
  return(c(delta, variance, deltaPercent, n))
}

calculateUtilities <- function(decisionTable, homePath, outcome, treatmentList){
  #the first feature starts at column 4, hence we extract the column names from 4th till the end.
  featureNames <- names(decisionTable)[ 4: (length(names(decisionTable))-4)]
  treatmentColumn <- c()
  cohortColumn <- c()
  utilityColumn <- c()
  for (treatment in treatmentList){
    inputData <- fread(paste0(homePath, treatment, 'SimulationDataTraining.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
    
    for (row in 1:nrow(decisionTable)) {
      treatmentColumn <- c(treatmentColumn, treatment)
      
      filteredData <- inputData
      ruleName <- as.character(decisionTable$rule[[row]])
      cohortColumn <- c(cohortColumn, ruleName)
      pcover <- decisionTable$pcover[[row]]
      for (i in 1:length(featureNames)) {
        feature <- featureNames[[i]]
        featureRange <- decisionTable[[feature]][[row]]
        filteredData <- subsetData(filteredData, featureRange, feature)
      }
      stats<- calculateStats(filteredData, outcome)
      utilityColumn <- c(utilityColumn, paste(stats[[1]], "; ", stats[[2]], "; ", pcover))
    }
  }
  resultDF <- data.frame("cohort" = cohortColumn, "treatment" = treatmentColumn, "utilities" = utilityColumn)
  return(resultDF)
}


calculateUtilitiesMergedTree <- function(decisionTableMerged, homePath, outcome, treatmentList){
  #the first feature starts at column 4, hence we extract the column names from 4th till the end.
  featureNames <- names(decisionTableMerged)[4:7]
  treatmentColumn <- c()
  cohortColumn <- c()
  utilityColumn <- c()
  pcoverList <- c()
  for (row in 1:nrow(decisionTableMerged)) {
    inputData <- fread(paste0(homePath, treatmentList[1], 'SimulationDataTraining.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
    filteredData <- inputData
    n.total <- nrow(inputData)
    for (i in 1:length(featureNames)) {
      feature <- featureNames[[i]]
      featureRange <- decisionTableMerged[[feature]][[row]]
      filteredData <- subsetData(filteredData, featureRange, feature)
    }
    stats<- calculateStats(filteredData, outcome)
    pcover <- as.double(stats[[4]])/n.total
    pcoverList <- c(pcoverList, pcover)
  }
  
  for (treatment in treatmentList){
    for (row in 1:nrow(decisionTableMerged)) {
      treatmentColumn <- c(treatmentColumn, treatment)
      ruleName <- paste("Rule number:", row)
      cohortColumn <- c(cohortColumn, ruleName)
      delta <- decisionTableMerged[[paste0("delta",treatment,outcome)]][[row]]
      variance <- decisionTableMerged[[paste0("variance",treatment,outcome)]][[row]]
      stats<- calculateStats(filteredData, outcome)
      pcover <- pcoverList[[row]]
      utilityColumn <- c(utilityColumn, paste(delta, "; ", variance, "; ", pcover))
    }
  }
  resultDF <- data.frame("cohort" = cohortColumn, "treatment" = treatmentColumn, "utilities" = utilityColumn)
  return(resultDF)
}

trimBucket <- function (x){
  # trim a string to remove white spaces
  # Args: x: the raw string.
  #
  # Returns:
  #   a trimmed string.
  gsub("\\)|\\]|\\(|\\[", "", x)
}


subsetData <- function(experimentData, featureRange, featureName){
  filteredData <- experimentData
  #if the rule is empty, return the whole dataset
  if (is.na(featureRange) || featureRange == "NA") return(filteredData)
  # else if it is a numerical feature, extract the upbound and lowerbound, and then apply filtering logic based on the rule
  else {
    featureBounds <-  unlist(strsplit(trimBucket(featureRange), ";"))
    lowerbound <- as.numeric(trimws(featureBounds[1]))
    upbound <- as.numeric(trimws(featureBounds[2]))
    
    rightInclusive <- F
    leftInclusive <- T
    
    filteredData <- subset(filteredData, 
       if (rightInclusive) filteredData[[featureName]] <= upbound else filteredData[[featureName]] < upbound
        &
      if (leftInclusive) filteredData[[featureName]]  >= lowerbound else filteredData[[featureName]] > lowerbound
    )
    return(filteredData)
  }
}


scoreTestDataGeneratePolicy <- function(decisionTable, homePath, treatmentList,  finalAssignment, method = "SGD"){
  featureNames <- names(decisionTable)[ 4: (length(names(decisionTable))-4)]
  outputData <- data.frame("ID" = as.numeric())
  for (treatment in treatmentList){
    outputData[[treatment]] <- as.numeric()
  }
  inputData <- fread(paste0(homePath, treatmentList[[1]], 'SimulationDataTest.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
  for (row in 1:nrow(decisionTable)) {
    filteredData <- inputData
    ruleName <- decisionTable$rule[[row]]
    cohortPolicy <- subset(finalAssignment, finalAssignment[["cohort"]] == ruleName)
    for (i in 1:length(featureNames)) {
      feature <- featureNames[[i]]
      featureRange <- decisionTable[[feature]][[row]]
      filteredData <- subsetData(filteredData, featureRange, feature)
    }
    if (nrow(filteredData) > 0){
      for (treatmentValue in treatmentList){
        subset.treatment <- subset(cohortPolicy, cohortPolicy[["treatment"]] == treatmentValue)
        if (nrow(subset.treatment) > 0){
          filteredData[[treatmentValue]] <- subset.treatment[[method]]
        }
        else{
          filteredData[[treatmentValue]] <- 0
        }
      }
      outputData <- rbind(outputData, subset(filteredData, select = c("ID", treatmentList)))
    }
  }
  return(outputData)
}


scoreTestDataGeneratePolicyMerged <- function(decisionTable, homePath, treatmentList,  finalAssignment, method = "SGD"){
  featureNames <- names(decisionTable)[ 4:7]
  outputData <- data.frame("ID" = as.numeric())
  for (treatment in treatmentList){
    outputData[[treatment]] <- as.numeric()
  }
  inputData <- fread(paste0(homePath, treatmentList[[1]], 'SimulationDataTest.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
  for (row in 1:nrow(decisionTable)) {
    filteredData <- inputData
    ruleName <- paste("Rule number:", row)
    cohortPolicy <- subset(finalAssignment, cohort == ruleName)
    for (i in 1:length(featureNames)) {
      feature <- featureNames[[i]]
      featureRange <- decisionTable[[feature]][[row]]
      filteredData <- subsetData(filteredData, featureRange, feature)
    }
    if (nrow(filteredData) > 0){
      for (treatmentValue in treatmentList){
        subset.treatment <- subset(cohortPolicy, cohortPolicy[["treatment"]] == treatmentValue)
        if (nrow(subset.treatment) > 0){
          filteredData[[treatmentValue]] <- subset.treatment[[method]]
        }
        else{
          filteredData[[treatmentValue]] <- 0
        }
      }
      outputData <- rbind(outputData, subset(filteredData, select = c("ID", treatmentList)))
    }
  }
  return(outputData)
}

########################### Helper Functions ##########################
stochasticOptimization <- function(homePath, utilityData, numCohorts, numTreatments, method, taus, alphas){
  ########################### Preparing Simulation Input Data ##########################
  l <- numTreatments # dimension of treatments
  N <- 250 #iters
  J <- 10 #number of samples used for evaluate the constraints
  m <- numCohorts # dimension of cohorts
  #utilityData <- utilityData[order(utilityData$cohort, utilityData$treatment),]
  utilityData <- data.frame(utilityData)
  d <- nrow(utilityData)
  
  metricDesiredDirections <- "up-up-up" #preferred metric directions for metric 1, 2, 3
  
  ########################### Running the Optimization per each cohort ##########################
  outputPath <- homePath #TODO: Please change the output location before running the simulation code.
  constraintIdx <- 1 # one of the important constraints that will be visualized in the result plot
  constraintDirection <- "up"
  numCores <- detectCores()
  # Extract the utility data (both point estimates and variances)
  metricDesiredDirectionsList <- strsplit(metricDesiredDirections, "-")
  muU <- unlist(processUtility(3, utilityData, 1)) * unlist(processUtility(3, utilityData, 3))
  sigmaUv <- processUtility(3, utilityData, 2) * processUtility(3, utilityData, 3) ^ 2
  sigmaU <- diag(as.list(sigmaUv))
  
  muVList <- list()
  sigmaVList <- list()
  
  for (i in 4 : ncol(utilityData)) {
    cnt <- i - 3
    metricDirection <- metricDesiredDirectionsList[[1]][cnt + 1]
    if (metricDirection == "up")
      muVList[[cnt]] <- processUtility(i, utilityData, 1) * processUtility(i, utilityData, 3)
    else if (metricDirection == "down")
      muVList[[cnt]] <- - processUtility(i, utilityData, 1) * processUtility(i, utilityData, 3)
    sigmaV <- processUtility(i, utilityData, 2) * processUtility(i, utilityData, 3) ^ 2
    sigmaVList[[cnt]] <- diag(as.list(sigmaV))
  }
  

  for (tau in taus) {
    for (alpha in alphas) {
      results <- list()
      registerDoParallel(numCores)
      # Parallelly run the stochastic optimizations
      results <- foreach (j = 1 : 10, .combine = "c") %dopar% {
        csa.solveFull(muU, sigmaU, muVList, sigmaVList, N, l, m, tau, alpha, J)
      }
      #stop parallel runs
      registerDoSEQ()
      
      csaSGDList <- list()
      csaAdagradList <- list()
      fxSGDList <- list()
      fxAdagradList <- list()
      fxSGDConstraintList <- list()
      fxAdagradConstraintList <- list()
      cnt <- 1
      for (i in seq(1, 20, by = 2)) {
        csaSGD <- results[[i]]
        csaAdagrad <- results[[i + 1]]
        csaSGDList[[cnt]] <- csaSGD
        csaAdagradList[[cnt]] <- csaAdagrad
        
        fxSGDList[[cnt]] <- unlist(lapply(csaSGD[[2]], function(x) { x %*% muU}))
        fxSGDConstraintList[[cnt]] <- unlist(lapply(csaSGD[[2]], function(x) { x %*% muVList[[constraintIdx]]}))
        fxAdagradList[[cnt]] <- unlist(lapply(csaAdagrad[[2]], function(x) { x %*% muU}))
        fxAdagradConstraintList[[cnt]] <- unlist(lapply(csaAdagrad[[2]], function(x) { x %*% muVList[[constraintIdx]]}))
        cnt <- cnt + 1
      }
      #extract summary stats for the results
      fxSGDMean <- summarizeResults(fxSGDList)[[1]]
      fxSGDStd <- summarizeResults(fxSGDList)[[2]]
      fxAdagradMean <- summarizeResults(fxAdagradList)[[1]]
      fxAdagradStd <- summarizeResults(fxAdagradList)[[2]]
      
      fxSGDConstraintMean <- summarizeResults(fxSGDConstraintList)[[1]]
      fxSGDConstraintStd <- summarizeResults(fxSGDConstraintList)[[2]]
      fxAdagradConstraintMean <- summarizeResults(fxAdagradConstraintList)[[1]]
      fxAdagradConstraintStd <- summarizeResults(fxAdagradConstraintList)[[2]]
      
      #save all the outputs into files
      resultDF <- data.frame("cohort" = utilityData$cohort, "treatment" = utilityData$treatment, "SGD" = csaSGD[[1]], "Adagrad" = csaAdagrad[[1]])
      #writeDetails(outputPath, csaSGD, muU, muVList, tau, alpha, metricDesiredDirectionsList, "SGDDetails")
      #writeDetails(outputPath, csaAdagrad, muU, muVList, tau, alpha, metricDesiredDirectionsList, "AdagradDetails")
      write.csv(resultDF, file = paste0(outputPath, '/FinalAssignments', method, tau, "-", alpha, '.csv'), row.names = FALSE, quote = FALSE)
      
      output <- paste0(outputPath, "/ObjVsOneConstraintPlot", tau, "-", alpha, ".pdf")
      plotResults(output, constraintDirection, fxSGDMean, fxAdagradMean, fxSGDStd, fxAdagradStd, fxSGDConstraintMean, fxAdagradConstraintMean, fxSGDConstraintStd, fxAdagradConstraintStd)
    }
  }
}

evaluateConstraints <- function(VList, xList, etaList, k, d, J, N){
  # evaluate k constriants E(-g(x,Vk)) <= etak and return the violated ones
  #
  # Args:
  #   VList: a list of values for the constraints.
  #   xList: a list represents the probabilistic assignments of treatments.
  #   etaList: a list of eta represent the thresholds of constraints.
  #   k: the number of constriants.
  #   d: dimension of x (number of treatment * number of cohorts)
  #   J: the number of samples to be evaluated
  #   N: the number of total iterations
  #
  # Returns:
  #   A list of index and a list of values for the constraints that got violated.
  VListSingle <- lapply(VList, function(x) { x[k,]})
  etaListSingle <- lapply(etaList, function(x) { x[k,]})
  
  randIdxs <- sample(1 : N, J, replace = TRUE)
  violateList <- list()
  idxList <- list()
  cnt <- 1
  sumV <- rep(0) * d
  for (i in 1 : length(etaList)) {
    muV <- colMeans(VList[[i]][randIdxs,])
    if (- muV %*% xList[[k]] > etaListSingle[[i]]) {
      violateList[[cnt]] <- as.matrix(VListSingle[[i]])
      idxList[[cnt]] <- i
      cnt <- cnt + 1
    }
  }
  list(idxList, violateList)
}

addCohortLevelConstraints <- function(A.matrix, l, m){
  # per each cohort add a column in matrix A
  #
  # Args:
  #   A.metrix: a matrix represent the constraints
  #   l: the number of treatments
  #   m: the numberof cohorts
  # Returns:
  #   a revised A matrix.
  baseCol <- rep(0, l * m)
  for (i in 0 : (m - 1)) {
    col <- baseCol
    col[c((l * i + 1) : (l * (i + 1)))] <- 1
    A.matrix <- rbind(A.matrix, t(col))
  }
  A.matrix
}


csa.solve <- function(muU, sigmaU, muVList, sigmaVList, N, l, m, gamma, etaList, alpha, s, J, type = "SGD") {
  # The code below solves Max x^tU s.t -x^Vk <= 0, 0 <= x <= 1, sum(x per cohort) = 1
  #
  # Args:
  #   muU: the mean of objective utility for each treatment and cohort.
  #   sigmaU: the variance of objective utility for each treatment and cohort.
  #   muVList: a list of the mean of constraint utilities for each treatment and cohort.
  #   sigmaVList: a list of the variance of constraint utilities for each treatment and cohort.
  #   N: number of iterations
  #   l: dimension of treatments
  #   m: dimension of cohorts
  #   gamma: initializations of the optimization.
  #   etaList: a list of eta represent the thresholds of constraints.
  #   alpha: hyperparameter of the learning rate.
  #   s: the starting iteration for evaluate whether the solution satisfy the constraints.
  #   J: the number of samples to be evaluated.
  #   type: the choice of optimization method "SGD" or "Adagrad" (default method is "SGD")
  # Returns:
  #   Results of the optimization in terms of
  # 1)final treatment assignment;
  # 2) treatment assignments for each iteration;
  # 3) the number of solutions that satisfy all the constraints.
  U <- mvrnorm(n = N, mu = muU, Sigma = sigmaU)
  VList <- list()
  for (i in 1 : length(sigmaVList)) {
    VList[[i]] <- mvrnorm(n = N, mu = muVList[[i]], Sigma = sigmaVList[[i]])
  }
  
  d = length(muU)
  xList <- list()
  xList[[1]] <- rep(0, d)
  
  if (type == "SGD") {
    # First we try regular Hk = I/alpha
    Hk <- 2 * sparseMatrix(i = 1 : d, j = 1 : d, x = rep(1, d)) / alpha
  }
  
  HkMat <- mat.or.vec(d, d)
  direction <- "0"
  
  for (k in 1 : N) {
    violations <- evaluateConstraints(VList, xList, etaList, k, d, J, N)
    violatedIdx <- violations[[1]]
    violatedList <- violations[[2]]
    if (length(violatedList) == 0) {
      gk <- - U[k,]
      direction <- "0"
    } else {
      # randomly choose a violated constriant as the gradient (gk)
      randIdx <- unlist(sample(1 : length(violatedList), 1, replace = FALSE))
      gk <- - as.matrix(violatedList[[randIdx]])
      direction <- as.character(violatedIdx[[randIdx]])
    }
    if (type == "Adagrad") {
      HkMat <- HkMat + gk %*% t(gk)
      Hk <- 2 * sparseMatrix(i = 1 : d, j = 1 : d, x = Re(sqrt(diag(HkMat))) / alpha)
    }
    results <- solve_osqp(P = Hk,
                          q = (gamma[k] * gk - Hk %*% xList[[k]]),
                          A = addCohortLevelConstraints(sparseMatrix(i = 1 : d, j = 1 : d, x = rep(1, d)), l, m),
                          l = c(rep(0, d), rep(0.999, m)),
                          u = c(rep(1, d), rep(1.001, m)),
                          osqpSettings(eps_abs = 1e-6, eps_rel = 1e-6, verbose = TRUE))
    xList[[k + 1]] <- results$x
  }
  
  numerator <- 0
  denominator <- 0
  count <- 0
  for (k in s : N) {
    violatedList <- evaluateConstraints(VList, xList, etaList, k, d, J, N)[[2]]
    if (length(violatedList) == 0) {
      numerator <- numerator + gamma[k] * xList[[k]]
      denominator <- denominator + gamma[k]
      count <- count + 1;
    }
  }
  
  xFinal <- numerator / denominator
  return(list(xFinal, xList, count))
}

csa.solveFull <- function(muU, sigmaU, muVList, sigmaVList, N, l, m, tau, alpha, J){
  # solve for the parameter assignments (Max x^tU s.t x^Vk >= 0, 0 <= x <= 1, sum(x) = 1)
  #
  # Args:
  #   muU: the mean of objective utility for each treatment and cohort.
  #   sigmaU: the variance of objective utility for each treatment and cohort.
  #   muVList: a list of the mean of constraint utilities for each treatment and cohort.
  #   sigmaVList: a list of the variance of constraint utilities for each treatment and cohort.
  #   N: number of iterations.
  #   l: dimension of treatments.
  #   m: dimension of cohorts.
  #   tau: hyperparameter for constraints violation level.
  #   alpha: hyperparameter of the learning rate.
  #   J: the number of samples to be evaluated.
  # Returns:
  #   the optimization solutions with SGD and Adagrad options.
  alpha <- 1
  s <- N / 2
  d <- l * m
  
  Dx <- sqrt(d) / alpha
  MF <- sqrt(sum(diag(sigmaU)) + sum(muU ^ 2))
  MGList <- list()
  for (i in 1 : length(muVList)) {
    MGList[[i]] <- sqrt(sum(diag(sigmaVList[[i]])) + sum(muVList[[i]] ^ 2))
  }
  
  gamma <- Dx / ((MF + sum(unlist(MGList))) * sqrt(1 : N))
  etaList <- list()
  for (i in 1 : length(MGList)) {
    etaList[[i]] <- as.matrix(tau * Dx * (MF + MGList[[i]]) / sqrt(1 : N))
  }
  
  csaSGD <- csa.solve(muU, sigmaU, muVList, sigmaVList, N, l, m, gamma, etaList, alpha, s, J, "SGD")
  csaAdagrad <- csa.solve(muU, sigmaU, muVList, sigmaVList, N, l, m, gamma, etaList, alpha, s, J, "Adagrad")
  list(csaSGD, csaAdagrad)
}

trim <- function (x){
  # trim a string to remove white spaces
  # Args: x: the raw string.
  #
  # Returns:
  #   a trimmed string.
  gsub("^\\s+|\\s+$", "", x)
}

processUtility <- function(colIdx, data, idx){
  # process the utility input data (mean pm error margin) and extract the mean & error margin information
  # Args:
  #   colIdx: column index of the utility in the dataframe.
  #   data: the dataframe with all utility information.
  #   idx: the index of the item within the column.
  # Returns:
  #   A matrix form of the processed utilities.
  utility <- c()
  for (i in 1 : length(data[, colIdx])) {
    utility[[i]] <- as.numeric(trim(strsplit(data[, colIdx][[i]], ";")[[1]][idx]))
  }
  as.matrix(utility)
}

writeDetails <- function(outputPath, result, muU, muVList, tau, alpha, metricDesiredDirectionsList, fileName) {
  # write details of the QP solutions in json format
  # Args:
  #   outputPath: the path of output directory.
  #   result: the result of solved treatment assignments from the optimization.
  #   muU: the mean of objective utility for each treatment and cohort.
  #   muVList: a list of the mean of constraint utilities for each treatment and cohort.
  #   tau: hyperparameter for constraints violation level.
  #   alpha: hyperparameter of the learning rate.
  #   metricDesiredDirectionsList: a list of the desired direction of metrics of interets
  #   fileName: the file name for the output.
  # Returns:
  # NULL
  detailResults <- list()
  detailResults[[1]] <- paste0("Objective Value: ", result[[1]] %*% muU)
  cnt <- 2
  for (i in 1 : length(muVList)) {
    metricDirection <- metricDesiredDirectionsList[[1]][i + 1]
    if (metricDirection == "up")
      detailResults[[cnt]] <- paste0("Constraint: ", result[[1]] %*% muVList[[i]])
    else detailResults[[cnt]] <- paste0("Constraint: ", - result[[1]] %*% muVList[[i]])
    cnt <- cnt + 1
  }
  detailResults[[cnt]] <- paste0("Chosen Output: ")
  detailResults[[cnt + 1]] <- result[[1]]
  detailResults[[cnt + 2]] <- paste0(" count:", result[[3]])
  jsonResults <- toJSON(detailResults, auto_unbox = TRUE)
  write(prettify(jsonResults), file = paste0(outputPath, '/', fileName, tau, "-", alpha))
}

plotResults <- function(output, constraintDirection, fxSGDMean, fxAdagradMean, fxSGDStd, fxAdagradStd, fxSGDConstraintMean, fxAdagradConstraintMean, fxSGDConstraintStd, fxAdagradConstraintStd){
  # plotting the Expectation of effect on objective metric along with the increasing iterations
  # Args:
  #   output: the path of the output.
  #   constraintDirection: the desired direction of the constraint.
  #   fxSGDMean: the mean of F(x,U) on the objective utility given the SGD solution.
  #   fxAdagradMean: the mean of F(x,U) on the objective utility given the Adagrad solution.
  #   fxSGDStd: the standard deviation of F(x,U) on the objective utility given the SGD solution.
  #   fxAdagradStd: the standard deviation of F(x,U) on the objective utility  given the Adagrad solution.
  #   fxSGDConstraintMean: the mean of F(x,V) on the constriant utility given the SGD solution.
  #   fxAdagradConstraintMean: the mean of F(x,V) on the constriant utility given the Adagrad solution.
  #   fxSGDConstraintStd: the standard deviation of F(x,V) on the constriant utility given the SGD solution.
  #   fxAdagradConstraintStd: the standard deviation of F(x,V) on the constriant utility given the Adagrad solution.
  # Returns:
  #   NULL
  x <- c(1 : length(fxSGDMean))
  if (constraintDirection == "down") {
    fxAdagradConstraintMean = - fxAdagradConstraintMean
  }
  
  plt <- ggplot() +
    geom_line(aes(x, fxSGDMean, colour = "SGD Objective"), size = 1.5) +
    geom_line(aes(x, fxAdagradMean, colour = "Adagrad Objective"), size = 1.5) +
    geom_ribbon(aes(x, ymin = fxSGDMean - fxSGDStd, ymax = fxSGDMean + fxSGDStd), fill = "#C77CFF", alpha = 0.3) +
    geom_ribbon(aes(x, ymin = fxAdagradMean - fxAdagradStd, ymax = fxAdagradMean + fxAdagradStd), fill = "#7CAE00", alpha = 0.3) +
    geom_line(aes(x, fxSGDConstraintMean, colour = "SGD Constraint"), size = 1.5) +
    geom_line(aes(x, fxAdagradConstraintMean, colour = "Adagrad Constraint"), size = 1.5) +
    geom_ribbon(aes(x, ymin = fxSGDConstraintMean - fxSGDConstraintStd, ymax = fxSGDConstraintMean + fxSGDConstraintStd), fill = "#00BFC4", alpha = 0.3) +
    geom_ribbon(aes(x, ymin = fxAdagradConstraintMean - fxAdagradConstraintStd, ymax = fxAdagradConstraintMean + fxAdagradConstraintStd), fill = "#F8766D", alpha = 0.3) +
    xlab("Iterations") +
    ylab("Estimations of Effects") +
    theme(text = element_text(size = 40),
          axis.text.x = element_text(size = 20),
          legend.title = element_blank(),
          legend.text = element_text(size = 20),
          legend.position = c(0.7, 0.4),
          legend.key.size = unit(4, 'lines'),
          axis.text.y = element_blank(), axis.ticks.y = element_blank())
  ggsave(output, plot = plt, width = 9, height = 7.5)
}


randPercent <- function(N, M) {
  # generate a list of N percentages that will sum up to a fixed values M
  # Args:
  #   N: the number of values in the output
  #   M: the number of the sum of the output values
  # Returns:
  #   A list of numeric values
  vec <- abs(rnorm(N))
  vec / sum(vec) * M
}

summarizeResults <- function(resultList) {
  # process the results (list format) and calculate the mean and standard deviations and output the summary stats
  # Args:
  #   resultList: A list of results
  #
  # Returns:
  #   A list with the mean and standard deviation of the results
  colSize = length(resultList)
  rowSize = length(resultList[[1]])
  resultMatrix <- matrix(unlist(resultList), ncol = colSize, nrow = rowSize)
  resultMean <- rowMeans(resultMatrix)
  resultStd <- unlist(apply(resultMatrix, 1, function(x) { sd(x)}))
  list(resultMean, resultStd)
}


# convert tree model to a set of decision rules and output a decision table (each row represent a rule)
convertToRules <- function(model, uniqueVariables, dataNew)
{
  if (! inherits(model, "rpart")) stop(print("Not a legitimate rpart tree"))
  statsDf <- data.frame("rule" = c(), "delta" = c(), "cover" = c(), "pcover" = c())
  if (length(uniqueVariables) == 0) statsDf
  else {
    for (i in 1 : length(uniqueVariables)) {
      statsDf[[unlist(uniqueVariables[i])]] <- NULL
    }
    rtree <- length(attr(model, "ylevels")) == 0
    target <- as.character(attr(model$terms, "variables")[2])
    frm <- model$frame
    names <- row.names(frm)
    ylevels <- attr(model, "ylevels")
    ds.size <- model$frame[1,]$n
    # convert each leaf node as a rule
    # Sort rules by coverage
    if (rtree)ordered <- rev(sort(frm$n, index = TRUE)$ix)
    # Sort rules by probabilty of second class (usually the last in binary class)
    else ordered <- rev(sort(frm$yval2[, 5], index = TRUE)$ix)
    for (i in ordered) {
      if (frm[i, 1] == "<leaf>") {
        if (rtree) delta <- frm[i,]$yval
        else delta <- ylevels[frm[i,]$yval]
        cover <- frm[i,]$n
        pcover <- (1.0 * cover) / ds.size
        pth <- rpart::path.rpart(model, nodes = as.numeric(names[i]), print.it = FALSE)
        pth <- unlist(pth)[- 1]
        if (! length(pth)){
          pth <- "True"
        }
        else {
          statsDf[i, "rule"] = paste("Rule number:", names[i])
          statsDf[i, "delta"] <- delta
          statsDf[i, "cover"] <- cover
          statsDf[i, "pcover"] <- pcover
          for (j in 1 : length(uniqueVariables)) {
            v <- uniqueVariables[[j]]
            factorFlag <- FALSE
            if (typeof(dataNew[[v]]) == "interger") {
              factorFlag <- TRUE
            }
            for (k in pth) {
              if (grepl(v, k)) {
                if (is.null(statsDf[i, v]) || is.na(statsDf[i, v]))
                  statsDf[i, v] <- equationToRule(k)
                else {
                  oldRule <- statsDf[i, v]
                  newRule <- equationToRule(k)
                  if (factorFlag)statsDf[i, v] <- extendFactorRule(oldRule, newRule)
                  else statsDf[i, v] <- extendNumericRule(oldRule, newRule)
                }
              }
            }
          }
        }
      }
    }
    statsDf[rowSums(is.na(statsDf)) != ncol(statsDf),]
  }
}

# convert an equation to a rule (e.g.: convert x1 > 0 to (0, double.max)
equationToRule <- function(equation){
  if (grepl("<", equation)) {
    if (grepl("=", equation)) {
      upbound = unlist(strsplit(equation, "<="))[2]
      paste("(", - .Machine$double.xmax, "; ", upbound, "]")
    }
    else {
      upbound = unlist(strsplit(equation, "<"))[2]
      paste("(", - .Machine$double.xmax, "; ", upbound, ")")
    }
  }
  else if (grepl(">", equation)) {
    if (grepl("=", equation)) {
      lowerbound <- unlist(strsplit(equation, ">="))[2]
      paste("[", lowerbound, "; ", .Machine$double.xmax , ")")
    }
    else {
      lowerbound <- unlist(strsplit(equation, ">"))[2]
      paste("(", lowerbound, "; ", .Machine$double.xmax , ")")
    }
  }
  else if (grepl("=", equation)) {
    items <- unlist(strsplit(equation, "="))[2]
    paste("(", gsub("\\,", "; ", items), ")")
  }
  else {
    stop(print("Not a legitimate equation"))
  }
}

# evaluate and compare two values (which are either lower bound or upper bound)
# first compare the numeric part of the bounds
# if the numeric parts are equal to each other, also check whether the bound is inclusive or exclusive
evaluate <- function(v1, v2){
  numV1 <- as.numeric(gsub("\\[|\\]|\\(|\\)", "", v1))
  numV2 <- as.numeric(gsub("\\[|\\]|\\(|\\)", "", v2))
  if (is.null(v1) || is.null(v2)) {
    NULL
  }
  else {
    if (trim(v1) == trim(v2))"="
    else if (numV1 < numV2)"<"
    else if (numV1 > numV2)">"
    else {
      if ((grepl("\\(", v1) && grepl("\\[", v2)) || (grepl("\\)", v1) && grepl("\\]", v2)))"<"
      else if ((grepl("\\[", v1) && grepl("\\(", v2)) || (grepl("\\]", v1) && grepl("\\)", v2)))">"
      else "="
    }
  }
}
# extend the catetgorical/factor rules (e.g.: merge rule 1: feature1 \in (us, uk) with rule 2: feature1 \in (us, bz))
extendFactorRule <- function(oldRule, newRule){
  oldRuleList <- unique(unlist(strsplit(gsub("\\(|\\)", "", oldRule), ";")))
  newRuleList <- unique(unlist(strsplit(gsub("\\(|\\)", "", newRule), ";")))
  combinedRule <- intersect(oldRuleList, newRuleList)
  paste0("(", paste(as.character(combinedRule), collapse = "; "), ")")
}
# extend the numeric rules (e.g.: merge rule 1: feature1 \belong (-inf, 0] with rule 2: feature1 \belong [-1, 8)
extendNumericRule <- function(oldRule, newRule){
  newRuleUpbound <- unlist(strsplit(newRule, ";"))[2]
  newRuleLowerbound <- unlist(strsplit(newRule, ";"))[1]
  oldRuleUpbound <- unlist(strsplit(oldRule, ";"))[2]
  oldRuleLowerbound <- unlist(strsplit(oldRule, ";"))[1]
  if ((evaluate(newRuleUpbound, oldRuleUpbound) == "<" || evaluate(newRuleUpbound, oldRuleUpbound) == "=") &&
      (evaluate(newRuleLowerbound, oldRuleLowerbound) == ">" || evaluate(newRuleLowerbound, oldRuleLowerbound) == "=")) {
    newRule
  }
  else if ((evaluate(newRuleUpbound, oldRuleUpbound) == ">" || evaluate(newRuleUpbound, oldRuleUpbound) == "=")
           &&
           (evaluate(newRuleLowerbound, oldRuleLowerbound) == "<" || evaluate(newRuleLowerbound, oldRuleLowerbound) == "=")) {
    oldRule
  }
  else if (evaluate(newRuleUpbound, oldRuleLowerbound) == ">" && evaluate(newRuleUpbound, oldRuleUpbound) == "<") {
    paste(oldRuleLowerbound, "; ", newRuleUpbound)
  }
  else if (evaluate(oldRuleUpbound, newRuleLowerbound) == ">" && evaluate(oldRuleUpbound, newRuleUpbound) == "<") {
    paste(newRuleLowerbound, "; ", oldRuleUpbound)
  }
  else NULL
}


evaluatePolicy <- function(policy, homePath, treatmentList, outcomeList){
  Y1.ATE <- 0
  Y2.ATE <- 0
  Y3.ATE <- 0
  for (treatment in treatmentList){
    policyW <- policy[[treatment]]
    testData <- fread(paste0(homePath, treatment, 'SimulationDataAllTest.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
    Y1.ATE <- Y1.ATE + mean(policyW * (testData[["Y1"]]-testData[["Y1.0"]]))/mean(testData[["Y1.0"]])
    Y2.ATE <- Y2.ATE + mean(policyW * (testData[["Y2"]]-testData[["Y2.0"]]))/mean(testData[["Y2.0"]])
    Y3.ATE <- Y3.ATE + mean(policyW * (testData[["Y3"]]-testData[["Y3.0"]]))/mean(testData[["Y3.0"]])
  }
  outputData <- data.frame("Y1" = c(Y1.ATE), "Y2" = c(Y2.ATE), "Y3" = c(Y3.ATE))
  return(outputData)
}

createHeuristCohorts <- function(featureNames, homePath, treatmentList){
  inputData <- fread(paste0(homePath, treatmentList[[1]], 'SimulationDataTraining.csv'), sep = ",", stringsAsFactors = FALSE, header = TRUE)
  n.total <- nrow(inputData)
  rule.n <- 1
  decisionTable <- data.frame("rule" = c(), "cover" = c(), "pcover" = c(), "H1" = c(), "H2" = c(), "H3" = c(), "H4" = c(), "delta" = c(), "deltaPercent" =c(), "pValueDeltaPercent" = c(), "variance" = c())

  list.cohorts <- list(c(0,0,0,0), c(1,0,0,0),c(1,1,0,0),c(1,1,1,0),
                       c(1,1,1,1), c(0,1,0,0),c(0,1,1,0), c(0,1,1,1),
                       c(1,0,0,1), c(0,0,1,0),c(0,0,1,1), c(1,1,0,1),
                       c(1,0,1,0), c(0,0,0,1),c(0,1,0,1), c(1,0,1,1))
  for (i in list.cohorts){
    decisionTable <- rbind(decisionTable, data.frame("rule" = c(paste("Rule number:", rule.n)), "cover" = c(NA), "pcover" = c(NA), "H1" = c(NA), "H2" = c(NA), "H3" = c(NA), "H4" = c(NA), "delta" = c(NA), "deltaPercent" =c(NA), "pValueDeltaPercent" = c(NA), "variance" = c(NA)))
    
    for (j in 1:length(featureNames)){
      feature <- featureNames[j]
      m <- median(inputData[[feature]])
      featureRange <- if (unlist(i)[[j]] == 0) paste(-Inf,";",m) else paste(m,";",Inf)
      decisionTable[[feature]][[rule.n]] <- featureRange
    }
    rule.n <- rule.n + 1
  }
  
  for (row in 1:nrow(decisionTable)) {

    filteredData <- inputData
    ruleName <- decisionTable$rule[[row]]
    pcover <- decisionTable$pcover[[row]]
    for (i in 1:length(featureNames)) {
      feature <- featureNames[[i]]
      featureRange <- decisionTable[[feature]][[row]]
      filteredData <- subsetData(filteredData, featureRange, feature)
    }
    stats<- calculateStats(filteredData, outcome)
    decisionTable[["rule"]][[row]] <- paste("Rule number:", row)
    decisionTable[["delta"]][[row]] <- stats[[1]]
    decisionTable[["cover"]][[row]] <- stats[[4]]
    decisionTable[["pcover"]][[row]] <- as.double(stats[[4]])/n.total
    
  }
  
  decisionTable <- decisionTable[, with(decisionTable,order(colnames(decisionTable))) ]
  decisionTable
}


