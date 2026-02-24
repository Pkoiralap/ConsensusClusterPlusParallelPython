# ConsensusClusterPlus: Algorithm and Implementation Guide

This document provides a deep dive into the **ConsensusClusterPlus** algorithm, covering its original R implementation, the architectural design of the Python port, and the advanced parallelization strategies used to scale it for large-scale genomic datasets.

---

## 1. The Core Algorithm: Consensus Clustering

Consensus Clustering, originally proposed by Monti et al. (2003), is an ensemble approach to unsupervised class discovery. Its primary goal is to determine the stability of clusters found in a dataset and to help identify the optimal number of clusters ($K$).

### 1.1 The Logic of Stability
If a dataset contains "true" underlying clusters, then subsampling the data should repeatedly yield the same cluster assignments for the same samples. If the clusters are artifacts of noise, subsampling will lead to inconsistent assignments.

### 1.2 Step-by-Step Mathematical Workflow

1.  **Input:** A data matrix $D$ (Features $	imes$ Items), a maximum number of clusters $K_{max}$, and a number of repetitions $R$ (typically 50–1000).
2.  **Subsampling:** For each repetition $r \in \{1...R\}$:
    *   Select a subset of items (samples) using proportion $pItem$ (default 0.8).
    *   Optionally select a subset of features (genes) using proportion $pFeature$.
3.  **Clustering:** Apply a base clustering algorithm (e.g., Hierarchical Clustering or KMeans) to the subsampled data for every $k \in \{2...K_{max}\}$.
4.  **Connectivity & Indicator Matrices:**
    *   **Connectivity Matrix ($M^{(r)}$):** An $N 	imes N$ matrix where $M_{ij} = 1$ if items $i$ and $j$ were in the same cluster in repetition $r$, and $0$ otherwise.
    *   **Indicator Matrix ($I^{(r)}$):** An $N 	imes N$ matrix where $I_{ij} = 1$ if items $i$ and $j$ were both present in the subsample for repetition $r$, and $0$ otherwise.
5.  **Consensus Matrix ($C$):** After $R$ repetitions, calculate the consensus matrix for each $k$:
    $$C_{ij} = \frac{\sum_r M_{ij}^{(r)}}{\sum_r I_{ij}^{(r)}}$$
    The value $C_{ij}$ represents the probability (between 0 and 1) that items $i$ and $j$ cluster together across all trials.
6.  **Final Clustering:** The Consensus Matrix $C$ is treated as a similarity matrix. The final cluster assignments for a given $k$ are determined by applying hierarchical clustering to $1 - C$ (the distance matrix).

---

## 2. The Original R Implementation

The R version (Bioconductor) is the "gold standard" for this method. 

### Key Features:
*   **Distance Metrics:** Supports `"pearson"`, `"spearman"`, and all standard `dist()` metrics.
*   **Visualizations:** Produces Heatmaps, CDF (Cumulative Distribution Function) plots, and Tracking plots.
*   **Hierarchical Engine:** Uses the `hclust` function.
*   **KMeans Engine:** Uses the Hartigan-Wong algorithm by default.

### The $K$ Selection Logic:
R evaluates the stability using the **CDF Plot**. A perfectly stable clustering would result in a Consensus Matrix where all values are either 0 or 1. The CDF curve of such a matrix would be a flat line at 0 and then a jump to 1. The "elbow" in the **Delta Area** plot (relative change in area under the CDF curve) is used to find the optimal $K$.

---

## 3. The Python Implementation

Our Python implementation reproduces the R logic while utilizing modern numerical libraries (`NumPy`, `SciPy`, `Scikit-Learn`) for speed.

### 3.1 Library Mappings
To ensure accuracy, we map R-specific methods to their Python equivalents:
*   **Distance:** `np.corrcoef` (Pearson) and `scipy.stats.spearmanr` (Spearman).
*   **Linkage:** `scipy.cluster.hierarchy.linkage` with a mapping for R-style names (e.g., `ward.D2` $	o$ `ward`).
*   **KMeans:** `sklearn.cluster.KMeans` with `k-means++` initialization for superior convergence.

### 3.2 Spearman Optimization
Spearman correlation is computationally expensive because it requires ranking data. Our implementation optimizes this by **pre-ranking** the entire dataset once if $pFeature=1.0$. This allows the worker threads to use the much faster Pearson correlation on the ranked data, reducing redundant calculations by $R 	imes$.

---

## 4. Parallelization & Optimization Strategies

A naive Python implementation of CCP is often slower than R due to the Global Interpreter Lock (GIL) and inefficient memory handling. We solved this using three key strategies:

### 4.1 Chunked Multiprocessing
Instead of submitting $R$ individual tasks to a process pool (which creates massive overhead for Inter-Process Communication), we split $R$ into **chunks** (e.g., 8 threads for 500 reps = ~62 reps per chunk). 
*   Each worker process handles a chunk internally.
*   Only the **aggregated** $M$ and $I$ matrices are returned to the main process, drastically reducing the data transfer volume.

### 4.2 Memory-Efficient Connectivity Updates
In the naive version, calculating the connectivity matrix $M$ involves checking every pair $(i, j)$, leading to $O(N^2)$ loops in Python, which are incredibly slow.
*   **Our Optimization:** For each cluster found in a subsample, we identify the indices of the members and perform a **vectorized slice update**:
    ```python
    ml_acc[k][np.ix_(cluster_items, cluster_items)] += 1
    ```
*   This uses highly optimized C-level NumPy routines to update the matrix in blocks, avoiding Python-level loops entirely.

### 4.3 Distance Matrix Pre-computation
If $pFeature=1.0$ (standard usage), the distance between samples $i$ and $j$ is constant across all repetitions. 
*   We calculate the **Global Distance Matrix** once in the main process.
*   We share this matrix with worker processes.
*   Workers simply slice the pre-computed matrix based on their subsampled items, eliminating thousands of redundant $O(Features 	imes Items^2)$ correlation calculations.

### 4.4 Resulting Performance
*   **Memory:** By using `int32` for accumulators and avoiding large temporary object creation, we keep the memory footprint stable regardless of thread count.
*   **Speed:** These optimizations allow the Python version to handle $10,000 	imes 2,000$ matrices in seconds, whereas the naive version would take hours or crash due to memory exhaustion.

---

## 5. Technical Nuances & R Equivalency

*   **Tie-Breaking:** Minor differences in floating-point precision between R (Fortran) and Python (C) can lead to different tie-breaking in hierarchical clustering.
*   **Zero Variance:** We implement robust checks for features with zero variance (which return `NaN` in correlations), filling them with 0 to prevent the clustering engine from crashing, mimicking R's error-handling behavior.

---

## 6. Algorithm's Complexity

Understanding the time complexity of Consensus Clustering is crucial for scaling it to large datasets. Let $N$ be the total number of items (samples), $M$ be the total number of features (genes), $R$ be the number of repetitions, and $K$ be the number of clusters evaluated. Let $N_{sub}$ and $M_{sub}$ be the subsampled items and features per repetition.

### 6.1 The Naive Complexity
A common misconception is that the algorithm scales linearly. However, the complexity is **quadratic** with respect to the number of items ($O(N^2)$), because clustering and distance metrics require comparing every sample to every other sample.

For each of the $R$ repetitions, the core operations are:
1.  **Distance Calculation:** $O(M_{sub} \times N_{sub}^2)$ — Calculating the pairwise distance between all subsampled items across all subsampled features.
2.  **Base Clustering:** $O(N_{sub}^2)$ — Building the hierarchical tree or fitting KMeans on the distance matrix.
3.  **Matrix Updates:** $O(K \times N_{sub}^2)$ — Updating the $N \times N$ connectivity matrix for every $k \in \{2...K_{max}\}$.

**Total Naive Complexity:** $\approx O(R \times N_{sub}^2 \times (M_{sub} + K))$

### 6.2 The Impact of Our Optimization
In the standard use case where we subsample items but *not* features ($pItem < 1.0, pFeature = 1.0$), the naive approach recalculates distances over all $M$ features $R$ times. 

Our Python implementation utilizes **Distance Matrix Pre-computation**. We calculate the global distance matrix exactly once before the repetitions begin. The worker threads then simply slice this pre-computed matrix based on the random subsamples for that repetition.

*   **Distance (One-time):** $O(M \times N^2)$
*   **Repetitions (R times):** $O(N_{sub}^2 \times K)$ (Clustering and Matrix Updates)

**Our Optimized Complexity:** $\approx O(M \times N^2) + O(R \times N_{sub}^2 \times K)$

By removing the feature dimension $M$ from the repetition loop, the optimization makes the algorithm exponentially faster. For a dataset with 10,000 features and 500 repetitions, bypassing the $O(R \times M \times N_{sub}^2)$ bottleneck prevents the algorithm from grinding to a halt.

### 6.3 Complexity Summary Table

| Step | Naive Complexity | Optimized Complexity | Reason |
| :--- | :--- | :--- | :--- |
| **Distance Calculation** | $O(R \times M \times N_{sub}^2)$ | $O(M \times N^2)$ | Computed once globally instead of $R$ times. |
| **Clustering (per rep)** | $O(R \times N_{sub}^2)$ | $O(R \times N_{sub}^2)$ | Tree building or KMeans fitting on subsamples. |
| **Connectivity Updates** | $O(R \times K \times N_{sub}^2)$ | $O(R \times K \times N_{sub}^2)$ | Updating the consensus votes for every $k$. |
| **Final Consensus Cut** | $O(N^2)$ | $O(N^2)$ | One final hierarchical clustering on the $N \times N$ consensus matrix. |
