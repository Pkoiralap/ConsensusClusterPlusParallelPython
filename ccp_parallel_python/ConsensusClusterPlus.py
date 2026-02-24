import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed

# Map R hclust methods to SciPy linkage methods
LINKAGE_MAP = {
    'ward.D': 'ward', 
    'ward.D2': 'ward',
    'single': 'single',
    'complete': 'complete',
    'average': 'average',
    'mcquitty': 'weighted',
    'median': 'median',
    'centroid': 'centroid'
}

def _cc_chunk_worker(chunk_seeds, d, full_dist_mat, maxK, pItem, pFeature, 
                     clusterAlg, distance, innerLinkage, 
                     weightsItem=None, weightsFeature=None):
    """
    Worker function that processes a chunk of subsampling repetitions.
    """
    innerLinkage = LINKAGE_MAP.get(innerLinkage, innerLinkage)
    
    if d is not None:
        n_features, n_items = d.shape
    elif full_dist_mat is not None:
        n_items = full_dist_mat.shape[0]
    else:
        raise ValueError("Either d or full_dist_mat must be provided to worker")
        
    mCount_acc = np.zeros((n_items, n_items), dtype=np.int32)
    ml_acc = {k: np.zeros((n_items, n_items), dtype=np.int32) for k in range(2, maxK + 1)}
    
    for seed in chunk_seeds:
        np.random.seed(seed)
        
        # 1. Subsample items
        n_sample_items = int(np.floor(n_items * pItem))
        sample_cols = np.sort(np.random.choice(n_items, n_sample_items, replace=False, p=weightsItem))
        
        # 2. Get distance matrix or data subset
        submat = None
        condensed_dist = None
        
        if pFeature < 1.0 or clusterAlg == 'km':
            if pFeature < 1.0:
                n_sample_features = int(np.floor(n_features * pFeature))
                sample_rows = np.sort(np.random.choice(n_features, n_sample_features, replace=False, p=weightsFeature))
                submat = d[sample_rows, :][:, sample_cols]
            else:
                submat = d[:, sample_cols]
            
            if clusterAlg != 'km':
                if distance == 'pearson':
                    # np.corrcoef is generally robust but handle zero variance
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = np.corrcoef(submat, rowvar=False)
                        corr = np.nan_to_num(corr, nan=0.0) # Match R's likely behavior or handle NAs
                    dist_mat = 1 - corr
                elif distance == 'spearman':
                    # If pFeature < 1.0, we must rank here
                    corr, _ = spearmanr(submat, axis=0)
                    if np.isscalar(corr): # Handle cases with very few items
                        corr = np.array([[1.0]])
                    corr = np.nan_to_num(corr, nan=0.0)
                    dist_mat = 1 - corr
                elif distance == 'euclidean':
                    dist_mat = squareform(pdist(submat.T, metric='euclidean'), checks=False)
                else:
                    raise ValueError(f"Unsupported distance: {distance}")
                
                np.fill_diagonal(dist_mat, 0)
                dist_mat = np.clip(dist_mat, 0, 2)
                condensed_dist = squareform(dist_mat, checks=False)
        else:
            # Use precomputed full_dist_mat
            sub_dist_mat = full_dist_mat[np.ix_(sample_cols, sample_cols)]
            condensed_dist = squareform(sub_dist_mat, checks=False)
            
        # 3. Cluster and Update
        mCount_acc[np.ix_(sample_cols, sample_cols)] += 1
        
        if clusterAlg == 'hc':
            # Check for Inf/NaN in condensed_dist which causes linkage to crash
            if not np.all(np.isfinite(condensed_dist)):
                warnings.warn("Non-finite distances encountered in subsample. Skipping.")
                continue
                
            Z = linkage(condensed_dist, method=innerLinkage)
            for k in range(2, maxK + 1):
                clusters = fcluster(Z, k, criterion='maxclust')
                for c in range(1, k + 1):
                    cluster_items = sample_cols[clusters == c]
                    if len(cluster_items) > 1:
                        ml_acc[k][np.ix_(cluster_items, cluster_items)] += 1
                
        elif clusterAlg == 'km':
            for k in range(2, maxK + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use 'k-means++' as requested by the user
                    km = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=10).fit(submat.T)
                clusters = km.labels_
                for c in np.unique(clusters):
                    cluster_items = sample_cols[clusters == c]
                    if len(cluster_items) > 1:
                        ml_acc[k][np.ix_(cluster_items, cluster_items)] += 1
                
    return mCount_acc, ml_acc

def ConsensusClusterPlus(d, maxK=3, reps=10, pItem=0.8, pFeature=1.0, 
                         clusterAlg="hc", innerLinkage="average", finalLinkage="average",
                         distance="pearson", seed=None, max_threads=1,
                         weightsItem=None, weightsFeature=None):
    """
    Optimized Python implementation of ConsensusClusterPlus with improved R compatibility.
    """
    if seed is None:
        seed = int(np.random.randint(0, 1e8))
        
    n_features, n_items = d.shape
    
    # Normalize weights if provided
    if weightsItem is not None:
        weightsItem = np.array(weightsItem, dtype=np.float64)
        weightsItem /= weightsItem.sum()
    if weightsFeature is not None:
        weightsFeature = np.array(weightsFeature, dtype=np.float64)
        weightsFeature /= weightsFeature.sum()

    # Pre-rank data for Spearman if pFeature == 1.0 to save time in workers
    working_d = d
    if distance == 'spearman' and pFeature == 1.0:
        # Spearman is Pearson on ranks. Pre-ranking columns:
        working_d = rankdata(d, axis=0)
        distance = 'pearson' # Workers can now use Pearson on the ranked data

    # Precompute full distance matrix if possible
    full_dist_mat = None
    if pFeature == 1.0 and clusterAlg != 'km':
        if distance == 'pearson':
            with np.errstate(divide='ignore', invalid='ignore'):
                full_dist_mat = 1 - np.corrcoef(working_d, rowvar=False)
                full_dist_mat = np.nan_to_num(full_dist_mat, nan=0.0)
        elif distance == 'spearman':
            # This branch is only taken if pFeature == 1.0 but ranking wasn't done above 
            # (which shouldn't happen with the optimization above)
            corr, _ = spearmanr(working_d, axis=0)
            full_dist_mat = 1 - np.nan_to_num(corr, nan=0.0)
        elif distance == 'euclidean':
            full_dist_mat = squareform(pdist(working_d.T, metric='euclidean'))
        
        if full_dist_mat is not None:
            np.fill_diagonal(full_dist_mat, 0)
            full_dist_mat = np.clip(full_dist_mat, 0, 2)
    
    mCount = np.zeros((n_items, n_items), dtype=np.int32)
    ml = {k: np.zeros((n_items, n_items), dtype=np.int32) for k in range(2, maxK + 1)}
    
    seeds = np.random.RandomState(seed).randint(0, int(1e8), size=reps)
    
    # Parallel execution with chunking
    if max_threads <= 1:
        mCount, ml = _cc_chunk_worker(seeds, working_d, full_dist_mat, maxK, pItem, pFeature, 
                                     clusterAlg, distance, innerLinkage,
                                     weightsItem, weightsFeature)
    else:
        num_workers = min(max_threads, reps)
        chunk_size = int(np.ceil(reps / num_workers))
        seeds_chunks = [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]
        
        # We only pass 'working_d' if it's needed in the worker
        data_to_pass = working_d if (pFeature < 1.0 or clusterAlg == 'km') else None
        
        with ProcessPoolExecutor(max_workers=len(seeds_chunks)) as executor:
            futures = [executor.submit(_cc_chunk_worker, sc, data_to_pass, full_dist_mat, 
                                       maxK, pItem, pFeature, clusterAlg, distance, 
                                       innerLinkage, weightsItem, weightsFeature) 
                       for sc in seeds_chunks]
            
            for future in as_completed(futures):
                mC, mL = future.result()
                mCount += mC
                for k in range(2, maxK + 1):
                    ml[k] += mL[k]
                    
    # Final consensus matrices and clustering
    res = {}
    finalLinkage_mapped = LINKAGE_MAP.get(finalLinkage, finalLinkage)
    
    for k in range(2, maxK + 1):
        consensus_matrix = np.zeros((n_items, n_items), dtype=np.float64)
        valid = mCount > 0
        consensus_matrix[valid] = ml[k][valid].astype(np.float64) / mCount[valid]
        
        # Symmetrize and set diagonal
        consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2
        np.fill_diagonal(consensus_matrix, 1.0)
        
        # Calculate distance from consensus matrix: dist = 1 - consensus
        dist_mat = 1 - consensus_matrix
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = np.clip(dist_mat, 0, 1)
        
        condensed_dist = squareform(dist_mat, checks=False)
        Z = linkage(condensed_dist, method=finalLinkage_mapped)
        clusters = fcluster(Z, k, criterion='maxclust')

        res[k] = {
            'consensusMatrix': consensus_matrix,
            'consensusClass': clusters
        }
        
    return res
