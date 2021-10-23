library(simcausal)

createCasualDAG <- function(seed, size, uncertaintyW, w1, w2, w3, w4, w5, w6, pow){
  D <- DAG.empty()
  D <- D +
    node("H1",
         distr = "rnorm",
         mean = 0.5,
         sd = 1) +      
    node("H2",
         distr = "rnorm",
         mean = (H1^2 * 2),
         sd = 2.5) +
    node("H3",
         distr = "rnorm",
         mean = ( 2 * H1 + 1.4 * H2),
         sd = 2) +
    node("H4",
         distr = "rnorm",
         mean = (-0.5 + 0.7 * H3 + 0.3 * H2),
         sd = 1
    ) +
    node("A",
         distr = "rbinom",
         prob = 1, size = 1) +
    node("U.Y",
         distr = "rnorm",
         mean = 3,
         sd = 1.5 * uncertaintyW) +
    node("Value",
         distr = "rnorm",
         mean = (U.Y + w1 * A * H2^pow + w2 * A * H3^pow + w3 * A * H4^pow), 
         sd = 0.5 * uncertaintyW) +
    node("Cost",
         distr = "runif",
         min= abs(w4 * A * H3^pow + U.Y + w5 * A * H4) * A / uncertaintyW,
         max = abs(w4 * A * H3^pow + U.Y + w5 * A * H4) * A 
    ) 
  Dset <- set.DAG(D, latent.v = c("U.Y"))
  
  A3 <- node("A", distr = "rbinom", prob = 1,size=3)
  Dset <- Dset + action("A3", nodes = A3)
  A2 <- node("A", distr = "rbinom", prob = 1,size=2)
  Dset <- Dset + action("A2", nodes = A2)
  A1 <- node("A", distr = "rbinom", prob = 1,size=1)
  Dset <- Dset + action("A1", nodes = A1)
  A0 <- node("A", distr = "rbinom", prob = 1,size=0)
  Dset <- Dset + action("A0", nodes = A0)
  
  plotDAG(Dset, xjitter = 0.3, yjitter = 0.04,
          edge_attrs = list(width = 0.5, arrow.width = 0.4, arrow.size = 0.8),
          vertex_attrs = list(size = 12, label.cex = 0.8))
  
  Xdat1 <- sim(DAG = Dset, actions = c("A3","A2","A1","A0"), n = size, rndseed = seed)
  
  Xdat1.A3 <- Xdat1[["A3"]]
  Xdat1.A2 <- Xdat1[["A2"]]
  Xdat1.A1 <- Xdat1[["A1"]]
  Xdat1.A0 <- Xdat1[["A0"]]
  
  Xdat1.A0["Control_value"] <- Xdat1.A0["Value"]
  Xdat1.A0["Control_cost"] <- 0
  
  Xdat1.A1["Control_value"] <- Xdat1.A0["Value"]
  Xdat1.A1["Control_cost"] <- 0
  
  Xdat1.A2["Control_value"] <- Xdat1.A0["Value"]
  Xdat1.A2["Control_cost"] <- 0
  
  Xdat1.A3["Control_value"] <- Xdat1.A0["Value"]
  Xdat1.A3["Control_cost"] <- 0
  
  return(list(rbind(Xdat1[["A3"]],Xdat1[["A2"]], Xdat1[["A1"]], Xdat1[["A0"]]),rbind(Xdat1.A3,Xdat1.A2,Xdat1.A1,Xdat1.A0))) 
}

generateSimulationData <- function(homePath, w1List, w2List, w3List, w4List, w5List, w6List, seed, uncertaintyW){
  seed <- seed * uncertaintyW
  simulationData <- createCasualDAG(seed, size, uncertaintyW, w1List[[1]], w2List[[1]], w3List[[1]], w4List[[1]], w5List[[1]], w6List[[1]], pow = 2)
  write.csv(simulationData[[2]], file = paste0(homePath,'/Data/Synthetic_data/', uncertaintyW, 'Weight_SimulationDataAllTraining.csv'), row.names = FALSE, quote = FALSE)
  simulationData <- createCasualDAG(seed+1, size, uncertaintyW, w1List[[2]], w2List[[2]], w3List[[2]], w4List[[2]], w5List[[2]], w6List[[2]], pow = 2)
  write.csv(simulationData[[2]], file = paste0(homePath,'/Data/Synthetic_data/', uncertaintyW, 'Weight_SimulationDataAllTest.csv'), row.names = FALSE, quote = FALSE)
}


homePath <- "/Users/atlasyu/Desktop/Supplementary_materials" #TODO: Please specify the home directory for the project


seed <- 12345

set.seed(seed)
size <- 20000
w1List <- c(rnorm(1, mean =0, sd = 1), rnorm(1, mean = 0, sd = 1), rnorm(1, mean = 0, sd = 1))
w2List <- c(rnorm(1, mean = 0, sd = 1), rnorm(1, mean = 0, sd = 1), rnorm(1, mean = 0, sd = 1))
w3List <-  c(rnorm(1, mean = 2, sd = 1), rnorm(1, mean = 2, sd = 1), rnorm(1, mean = 2, sd = 1))
w4List <-  c(rnorm(1, mean = 2, sd = 1), rnorm(1, mean = 2, sd = 1), rnorm(1, mean = 2, sd = 1))
w5List <-  c(rnorm(1, mean = 4, sd = 1), rnorm(1, mean = 4, sd = 1), rnorm(1, mean = 4, sd = 1))
w6List <-   c(rnorm(1, mean = 4, sd = 1), rnorm(1, mean = 4, sd = 1), rnorm(1, mean = 4, sd = 1))

uncertaintyWList <- c(5,10,15,20)

for (uncertaintyW in uncertaintyWList){
  generateSimulationData(homePath, w1List, w2List, w3List, w4List, w5List, w6List, seed, uncertaintyW)
}