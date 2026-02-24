import numpy as np
import pandas as pd
import os
import shutil
import datetime

def create_dataset():
    # Ask for x and y dynamically
    try:
        x = int(input("Enter number of features (x): ") or "1000")
        y = int(input("Enter number of samples (y): ") or "100")
    except ValueError:
        print("Invalid input. Using defaults: x=1000, y=100")
        x, y = 1000, 100

    np.random.seed(42)
    data = np.random.rand(x, y)
    
    # Introduce 10 clusters
    n_clusters = 10
    samples_per_cluster = y // n_clusters
    if samples_per_cluster > 0:
        for i in range(n_clusters):
            start_col = i * samples_per_cluster
            end_col = (i + 1) * samples_per_cluster if i < n_clusters - 1 else y
            
            # Add signal to a unique subset of rows for each cluster to make them distinct
            rows_per_cluster = x // n_clusters
            if rows_per_cluster > 0:
                start_row = i * rows_per_cluster
                end_row = (i + 1) * rows_per_cluster
                data[start_row:end_row, start_col:end_col] += 3.0

    # Create experiment directory structure
    experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", experiment_id)
    python_dir = os.path.join(exp_dir, "python")
    r_dir = os.path.join(exp_dir, "r")
    
    os.makedirs(python_dir, exist_ok=True)
    os.makedirs(r_dir, exist_ok=True)

    # Save data
    df = pd.DataFrame(data, 
                      index=[f"gene_{i}" for i in range(x)],
                      columns=[f"sample_{i}" for i in range(y)])
    data_path = os.path.join(exp_dir, "data.csv")
    df.to_csv(data_path)
    print(f"Created {data_path} with shape x: {x}, y: {y}")

    # Copy implementation and run scripts
    # Python files
    shutil.copy("python/ConsensusClusterPlus.py", python_dir)
    shutil.copy("python/run_ccp.py", python_dir)
    
    # R files
    shutil.copy("R/ConsensusClusterPlus.R", r_dir)
    shutil.copy("python/run_ccp.R", r_dir)

    # Comparison script to experiment root
    shutil.copy("python/compare_results.py", exp_dir)

    print(f"Experiment setup complete in: {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    create_dataset()
