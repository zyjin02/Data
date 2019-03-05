import random
import sys
import math
import pandas as pd
import numpy  as np
import copy   as copy
"""
K-means clustering (Question)
"""


def myKMeans(dataset, k = 3, seed = 42):
    """Returns the centroids of k clusters for the input dataset.
    Parameters
    ----------
    dataset: a pandas DataFrame
    k: number of clusters
    seed: a random state to make the randomness deterministic
    
    Examples
    ----------
    myKMeans(df, 5, 123)
    myKMeans(df)
    
    Notes
    ----------
    The centroids are returned as a new pandas DataFrame with k rows.
    
    """
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy = getNumFeatures(dataset)
    old_ctrd = getInitialCentroids(dataset_copy, k, seed)
    run_count = 0
    while(1):
        print("run_count = %d" %run_count)
        tmp_label =  [getLabels(dataset_copy, old_ctrd)]
        new_ctrd = computeCentroids(dataset_copy, np.unique(tmp_label))
        new_ctrd = new_ctrd.drop(columns = 'Cluster')
        if (stopClustering(old_ctrd, new_ctrd, run_count)) :
            break
        old_ctrd = new_ctrd
        run_count = run_count + 1
    return old_ctrd
        
def getNumFeatures(dataset):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dataset = dataset.select_dtypes(include=numerics)
    return dataset
    """Returns a dataset with only numerical columns in the original dataset"""


def getInitialCentroids(dataset, k, seed):
    random.seed(seed)
    df_ctrd = pd.DataFrame(np.random.randn(k, len(dataset.columns)))
    df_ctrd.columns = dataset.columns
    return(df_ctrd.sort_values(by = df_ctrd.columns[0]))
        
    """Returns k randomly selected initial centroids of the dataset"""


def getLabels(dataset, centroids):
    tmp_label = np.empty(dataset.shape[0], dtype=int)
    for i in range(dataset.shape[0]):
        tmp_min_distance = sys.float_info.max
        for j in range(centroids.shape[0]):
            tmp_d = computeDistance(pd.DataFrame([dataset.iloc[i]]), pd.DataFrame([centroids.iloc[j]]))
            if (tmp_d < tmp_min_distance) :
                tmp_min_distance = tmp_d
                tmp_label[i] = str(int(j))
    dataset['Cluster'] = pd.DataFrame(tmp_label)
    return tmp_label
    """Assigns labels (i.e. 0 to k-1) to individual instances in the dataset.
    Each instance is assigned to its nearest centroid.
    """    


def computeCentroids(dataset, labels):
    centroids_df = pd.DataFrame(columns = dataset.columns)
    for x in labels:
        tmp_df = dataset[dataset['Cluster'] == x]
        tmp_ct = [tmp_df.mean()]
        centroids_df = centroids_df.append(pd.DataFrame(tmp_ct),  ignore_index=True)
    return(centroids_df.sort_values(by = centroids_df.columns[0]))
    """Returns the centroids of individual groups, defined by labels, in the dataset"""
    
def stopClustering(oldCentroids, newCentroids, numIterations, maxNumIterations = 100, tol = 1e-4):
    if (numIterations == maxNumIterations): return True
    elif (computeDistance(oldCentroids, newCentroids) < tol): return True
    else: return False
    """Returns a boolean value determining whether the k-means clustering converged.
    Two stopping criteria: 
    (1) The distance between the old and new centroids is within tolerance OR
    (2) The maximum number of iterations is reached 
    """
    
def computeDistance(orig, dest):
    d_sum = 0
    for i in range(orig.shape[0]):
        df_sq = (pd.DataFrame([orig.iloc[i] - dest.iloc[i]])) ** 2
        d_sum = d_sum + math.sqrt(df_sq.sum(axis = 1))       
    return(d_sum)

        
        
        
        
        
        
        
        
        