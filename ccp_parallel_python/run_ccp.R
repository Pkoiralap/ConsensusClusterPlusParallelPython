#!/usr/bin/env Rscript

# Load original ConsensusClusterPlus
if (file.exists("ConsensusClusterPlus.R")) {
  source("ConsensusClusterPlus.R")
} else {
  source("../../R/ConsensusClusterPlus.R")
}

# Suppress warnings for clean output
options(warn=-1)

data_file <- "../data.csv"
if (!file.exists(data_file)) {
  data_file <- "test_data.csv"
}

data <- read.csv(data_file, row.names=1)
data_mat <- as.matrix(data)

cat("Running R ConsensusClusterPlus...
")
start_time <- Sys.time()
maxK <- 12
res <- ConsensusClusterPlus(d=data_mat, maxK=maxK, reps=50, pItem=0.8, pFeature=1, 
                            clusterAlg="hc", distance="pearson", seed=123, plot=NULL)
end_time <- Sys.time()
print(end_time - start_time)

for (k in 2:maxK) {
  # Save consensus matrix
  write.table(res[[k]]$consensusMatrix, file=paste0("r_consensusMatrix_k", k, ".csv"), sep=",", row.names=FALSE, col.names=FALSE)
  
  # Save consensus class
  df_class <- data.frame(item=names(res[[k]]$consensusClass), class=as.numeric(res[[k]]$consensusClass))
  write.csv(df_class, file=paste0("r_consensusClass_k", k, ".csv"), row.names=FALSE)
}
cat("R run complete.
")
