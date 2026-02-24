#!/usr/bin/env python3
import pandas as pd
import numpy as np
from ConsensusClusterPlus import ConsensusClusterPlus
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run ConsensusClusterPlus in Python")
    parser.add_argument("--threads", type=int, default=64, help="Number of threads/processes to use")
    args = parser.parse_args()
    
    data_file = "../data.csv"
    if not os.path.exists(data_file):
        data_file = "test_data.csv" # fallback for old usage
    
    df = pd.read_csv(data_file, index_col=0)
    data = df.values
    
    print(f"Running Python ConsensusClusterPlus with {args.threads} threads...")
    start_time = time.time()
    # Using maxK=12 to ensure we can see the 10 clusters
    res = ConsensusClusterPlus(data, maxK=12, reps=500, pItem=0.8, pFeature=1.0, 
                               clusterAlg="hc", distance="pearson", max_threads=args.threads, seed=123)
    end_time = time.time()
    print(f"Python run complete in {end_time - start_time:.2f} seconds.")
    
    for k in res.keys():
        np.savetxt(f"py_consensusMatrix_k{k}.csv", res[k]['consensusMatrix'], delimiter=",")
        
        df_class = pd.DataFrame({'item': df.columns, 'class': res[k]['consensusClass']})
        df_class.to_csv(f"py_consensusClass_k{k}.csv", index=False)

if __name__ == "__main__":
    main()
