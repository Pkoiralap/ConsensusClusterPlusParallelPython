#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import os

def compare():
    print("Comparing Python and R ConsensusClusterPlus results...")
    maxK = 12
    for k in range(2, maxK + 1):
        py_mat_file = os.path.join("python", f"py_consensusMatrix_k{k}.csv")
        r_mat_file = os.path.join("r", f"r_consensusMatrix_k{k}.csv")
        py_cls_file = os.path.join("python", f"py_consensusClass_k{k}.csv")
        r_cls_file = os.path.join("r", f"r_consensusClass_k{k}.csv")
        
        if not (os.path.exists(py_mat_file) and os.path.exists(r_mat_file)):
            # Try looking in current directory as fallback
            py_mat_file = f"py_consensusMatrix_k{k}.csv"
            r_mat_file = f"r_consensusMatrix_k{k}.csv"
            py_cls_file = f"py_consensusClass_k{k}.csv"
            r_cls_file = f"r_consensusClass_k{k}.csv"
            
            if not (os.path.exists(py_mat_file) and os.path.exists(r_mat_file)):
                print(f"Missing results for k={k}. Skipping.")
                continue
            
        py_mat = np.loadtxt(py_mat_file, delimiter=",")
        r_mat = np.loadtxt(r_mat_file, delimiter=",")
        
        # Mean absolute difference between consensus matrices
        diff = np.abs(py_mat - r_mat).mean()
        max_diff = np.abs(py_mat - r_mat).max()
        print(f"\n--- k={k} ---")
        print(f"Consensus Matrix Differences:")
        print(f"  Mean Absolute Error: {diff:.4f}")
        print(f"  Max Absolute Error:  {max_diff:.4f}")
        
        # Compare class assignments using Adjusted Rand Index
        py_cls = pd.read_csv(py_cls_file)['class'].values
        r_cls = pd.read_csv(r_cls_file)['class'].values
        
        ari = adjusted_rand_score(py_cls, r_cls)
        print(f"Class Assignments (Adjusted Rand Index): {ari:.4f} (1.0 = perfect match)")

if __name__ == "__main__":
    compare()
