# We are trying to merging multiple trees into one single cohort definition
unionSeparator <- " or "
rangeSeparator <- ";"
defaultRange <- "-Inf; Inf"
########################### Helper Functions ##########################
trim <- function (x){
  # trim a string to remove white spaces
  # Args: x: the raw string
  #
  # Returns:
  #   a trimmed string.
  gsub("^\\s+|\\s+$", "", x)
}

removeBuckets <- function(str){
  gsub("\\[|\\]|\\(|\\)", "", str)
}

extractRange <- function(str){
  # extract the string to output a set of numeric values
  # Args:
  #   str: the raw string
  # Returns:
  #   A set of numeric values.
  str <- removeBuckets(str)
  range.list <- list()
  
  if (str == "" || is.na(str))  range.list <- list(c(-Inf, Inf))
  else if (grepl("or", str, fixed = TRUE)){
    str.list <- unlist(strsplit(str, unionSeparator))
    for (idx in 1:length(str.list)){
      range.list[[idx]] <- as.numeric(trim(unlist(strsplit(str.list[[idx]], rangeSeparator))))
    }
  }
  else range.list <- list(as.numeric(trim(unlist(strsplit(str, rangeSeparator)))))
  range.list
}

intersectRange <- function(range.1, range.2){
  range.1.lo = range.1[1]
  range.1.hi = range.1[2]
  range.2.lo = range.2[1]
  range.2.hi = range.2[2]
  
  if (range.1.lo >= range.2.hi || range.2.lo >= range.1.hi) ""
  else paste(as.character(max(range.1.lo, range.2.lo)), rangeSeparator, as.character(min(range.1.hi, range.2.hi)))
}

intersectFeatureRanges <-function(ranges.1, ranges.2){
  outputRange = ""
  ranges.1.list = extractRange(ranges.1)
  ranges.2.list = extractRange(ranges.2)
  for (range.1 in ranges.1.list){
    for (range.2 in ranges.2.list){
      if (intersectRange(range.1, range.2) != ""){
        outputRange = if (outputRange == "") intersectRange(range.1, range.2) else outputRange + unionSeparator + intersectRange(range.1, range.2)
      }
    }
  }
  outputRange
}

intersectSegment <- function(S.1, S.2, l){
  # take intersection of two Segments by evaluate the sets per feature.
  # Args:
  #   S.1: the first segment.
  #   S.2: the second segment.
  # Returns:
  #   Segment.out: the intersect of S.1 and S.2.
  Segment.out <- list()

  S.1.sub <- S.1[, c(4:7)]
  S.2.sub <- S.2[, c(4:7)]
  
  for (i in 1 : ncol(S.1.sub)) {
    feature.range.1 <- S.1.sub[[i]]
    feature.range.2 <- if (i > length(S.2.sub)) defaultRange else S.2.sub[[i]]
    range.intersect <- intersectFeatureRanges(feature.range.1 , feature.range.2)

    Segment.out[[i]] <- range.intersect
  }

  S.out <- data.frame(S.1)
  S.out[["H1"]] <- Segment.out[[1]]
  S.out[["H2"]] <- Segment.out[[2]]
  S.out[["H3"]] <- Segment.out[[3]]
  S.out[["H4"]] <- Segment.out[[4]]
  
  S.out <- updateEstimations(S.out, S.1, S.2, l)
  #S.out <- expandSegment(data.frame(S.out))
  S.out
}

expandSegment <- function(S.input){
  S.input.sub <- S.input[, c(4:7)]
  S.out <- S.input[0, ]
  for (i in 1 : ncol(S.input.sub)) {
    feature.range <- S.input.sub[[i]]
    if (grepl("or", feature.range)){
      feature.range.list <- unlist(strsplit(feature.range, unionSeparator))
      for (j in 1:length(feature.range.list)){
        S.add <- S.input
        S.add[[paste0("H", i)]] <- feature.range.list[[j]]
        S.out <- rbind(S.out, S.add)
      }
    }
  }
  if (nrow(S.out) == 0) S.input else data.frame(S.out)
}

updateEstimations <- function(S.out, S.1, S.2, l){
  t.idx <- ceiling((l-1)/length(outcomeList))
  o.idx <- if ((l-1)%%length(outcomeList) == 0) length(outcomeList) else (l-1)%%length(outcomeList)
  
  delta.name.1 <- paste0("delta", treatmentList[t.idx], outcomeList[o.idx])
  variance.name.1 <- paste0("variance", treatmentList[t.idx], outcomeList[o.idx])
  
  S.out[[delta.name.1]] <- if (l-1 == 1)  S.1[["delta"]] else S.1[[delta.name.1]]
  S.out[[variance.name.1]] <- if (l-1 == 1)  S.1[["variance"]] else S.1[[variance.name.1]]
  
  t.idx.2 <- ceiling(l/length(outcomeList))
  o.idx.2 <- if (l%%length(outcomeList) == 0) length(outcomeList) else l%%length(outcomeList)
  
  delta.name.2 <- paste0("delta", treatmentList[t.idx.2], outcomeList[o.idx.2])
  variance.name.2 <- paste0("variance", treatmentList[t.idx.2], outcomeList[o.idx.2])
  
  S.out[[delta.name.2]] <- S.2[["delta"]]
  S.out[[variance.name.2]] <- S.2[["variance"]]
  
  S.out
}


getValidRange <- function(lo, hi){
  if (lo >= hi) "" else paste(as.character(lo), rangeSeparator, as.character(hi))
}

complementRange <- function(range.base, range.subtract){
  range.base.lo = range.base[1]
  range.base.hi = range.base[2]
  range.subtract.lo = range.subtract[1]
  range.subtract.hi = range.subtract[2]
  
  if (range.base.lo >= range.subtract.hi || range.subtract.lo >= range.base.hi) getValidRange(range.base.lo, range.base.hi)
  else{
    output.range.1 <- getValidRange(range.base.lo, range.subtract.lo)
    output.range.2 <- getValidRange(range.subtract.hi, range.base.hi)
    if (output.range.1 != "" && output.range.2 != "") paste(output.range.1, unionSeparator, output.range.2)
    else if (output.range.1 != "") output.range.1
    else if (output.range.2 != "") output.range.2
    else ""
  } 
}

complementFeatureRange <-function(ranges.1, ranges.2){
  outputRange <- ""
  ranges.1.list = extractRange(ranges.1)
  ranges.2.list = extractRange(ranges.2)
  for (range.1 in ranges.1.list){
    outputRange.base <- ""
    for (range.2 in ranges.2.list){
      if (complementRange(range.1, range.2) != ""){
        outputRange.base <- if (outputRange.base == "") complementRange(range.1, range.2) else intersectFeatureRanges(outputRange.base, complementRange(range.1, range.2))
      }
    }
    outputRange <- if (outputRange == "") outputRange.base else paste(outputRange, unionSeparator, outputRange.base)
  }
  outputRange
}

removeSpaces <- function(str){
  gsub(" ", "", str)
}

equalRange <- function(range.1, range.2){
  range.1.list <- extractRange(range.1)[[1]]
  range.2.list <- extractRange(range.2)[[1]]
  if (length(range.1.list) != length(range.2.list)) return(FALSE)
  else{
    for(i in 1:length(range.1.list)){
      if (range.1.list[1] != range.2.list[1] || range.1.list[2] != range.2.list[2]) return(FALSE)
    }
    return(TRUE)
  }
} 

complementSegment <- function(S.1, S.2, l){
  # Get (S.1 - S.2) by evaluate the sets per feature
  # Args:
  #   S.1: the first segment
  #   S.2: the second segment
  # Returns:
  #   Segment.out: (S.1 - S.2).
  Segment.out <- list()

  S.1.sub <- S.1[, c(4:7)]
  S.2.sub <- S.2[, c(4:7)]
  
  if (equalSegment(S.1, S.2)){
    S.out <- S.1[0, ]
  }
  else{
    for (i in 1 : ncol(S.1.sub)) {
      feature.range.1 <- S.1.sub[[i]]
      feature.range.2 <- if (i > length(S.2.sub)) defaultRange else S.2.sub[[i]]
      
      if (!equalRange(feature.range.1, feature.range.2)){
        range.complement <- complementFeatureRange(feature.range.1, feature.range.2)
        Segment.out[[i]] <- range.complement
      }
      else{
        Segment.out[[i]] <- feature.range.1
      } 
    }
    
    S.out <- S.1
  
    S.out[["H1"]] <- Segment.out[[1]]
    S.out[["H2"]] <- Segment.out[[2]]
    S.out[["H3"]] <- Segment.out[[3]]
    S.out[["H4"]] <- Segment.out[[4]]
  
    S.out <- expandSegment(data.frame(S.out))
    S.out <- updateEstimations(S.out, S.1, S.2, l)
  }
  S.out
}

emptySegment <- function(S){
  # evaluate whether a segment is empty
  # Args:
  #   S: the input segment.
  # Returns:
  #   flag: a boolean represent whether the segment is empty
  S <-data.frame(S)
  size <- nrow(S)
  if (size == 0) return(TRUE)
  else{
    for (i in 4:7) {
      if (!is.na(S[,i]) && S[,i] == "") return(TRUE)
    }
    return(FALSE)
  }
}

equalSegment <- function(S.1, S.2){
  # evaluate whether two segments are equal
  # Args:
  #   S.1: the first segment.
  #   S.2: the second segment.
  # Returns:
  #   flag: a boolean represent whether the two segments are equal.
  flag <- TRUE
  for (i in 4:7) {
    range.1 <- S.1[[i]]
    range.2 <- S.2[[i]]
    if (!equalRange(range.1, range.2)) flag <- FALSE
  }
  flag
}

replaceSegments <- function(C, Segments.remove, Segments.add){
  # replace a segment with a list of other segments
  # Args:
  #   C: the decision table (a list of Segments) need to be modified.
  #   Segments.remove: the segment to be removed.
  #   Segments.add: the segments to be added.
  # Returns:
  #   C.out: the revised decision table (a list of Segments).
  C.out <- C[0,]
  for (row in 1:nrow(C)) {
    Seg <- C[row,]
    
    
    if (!equalSegment(Seg, Segments.remove)) {
      C.out <- rbind(C.out, Seg)
    }
  }
  for (seg in Segments.add) {
    if(!emptySegment(seg)){
      C.out <- rbind(C.out, seg)
    }
  }
  C.out
}

mergeTrees <- function(dt.list){
  # merge trees in a sequential manner
  # Args:
  #   Cohorts: the Trees for mereging in the form of decision table (list of Segments) formats.
  #   Metrics: the dataframe with unit level metric data and treatment indicator variable.
  # Returns:
  #   C.out: the final mereged cohort definition.
  C.out <- data.frame(dt.list[[1]])
  
  for (l in 2 : length(dt.list)) {
    print(l)
    C.base <- C.out
    for (row.base in 1:nrow(C.base)) {
      S.base <- C.base[row.base,]
      for (row.cross in 1:nrow(dt.list[[l]])) {
        S.cross <- dt.list[[l]][row.cross,]
        if (!emptySegment(S.base)){
          S.int <- intersectSegment(S.base, S.cross, l)
          if (!emptySegment(S.int)) {
            if (!equalSegment(S.base, S.int)){
              S.complement <- complementSegment(S.base, S.int, l)
              C.out <- replaceSegments(C.out, S.base, list(S.int, S.complement))
              S.base <- S.complement
            }
            else{
              C.out <- replaceSegments(C.out, S.base, list(S.int))
            }
          }
        }
      }
    }
  }
  data.frame(C.out)
}

imputeCols <- function(dt.list){
  dt.out.list <- list()
  for (i in 1:length(dt.list)){
    dt <- dt.list[i]
    dt.out <- data.frame(dt)
    for (featureCol in c("H1", "H2", "H3", "H4")){
        if (!featureCol %in% colnames(dt.out)){
          dt.out[[featureCol]] <- NA
        }
    }
    dt.out <- dt.out[, with(dt.out,order(colnames(dt.out))) ]
    for (treatment in treatmentList){
      for (outcome in outcomeList){
        dt.out[[paste0("delta",treatment, outcome)]] <- NA
        dt.out[[paste0("variance",treatment, outcome)]] <- NA
      }
    }
    dt.out.list[[i]] <- dt.out
  }
  dt.out.list
}

########################### Preparing Example Input Data ##########################
#Tree models in the forms of Cohorts
#read in decision tables
dt.1 <- fread(file = '/Users/yunz/Code/prophet/A2Y1decisionTable.csv', header = TRUE )
dt.2 <- fread(file = '/Users/yunz/Code/prophet/A3Y1decisionTable.csv', header = TRUE )

dt.list <- list(dt.1, dt.2)
#imput data

dt.out.list <- imputeCols(dt.list)
S.1 <- dt.out.list[[1]][1,]
S.2 <- dt.out.list[[2]][1,]

#dt.out.list <- list(S.1 , dt.out.list[[2]])

########################### Running the Tree Merge Algorithm ##########################

dt.output <- mergeTrees(dt.out.list)
