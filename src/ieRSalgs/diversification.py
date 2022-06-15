'''
Diversification alg:
    Input: 
        candidates: Top N (=200) recommendations, ['item', 'prediction'] or ['item', 'score'] or ['item', 'discounted_score']
                    or ['item', 'joy', 'trust', 'fear', 'surprise', 'sadness', 'digust', 'anger', 'anticipation']
        vectors: np.ndarray, latent feature/ emotional signature of the full offline items, with movie_id indexed/labeled
        items: np.ndarray, row indices of vectors, also = itemIDs
        numRecs: int
    Output:
        Diversified top-N (=20) with item_id
'''
import pandas as pd
import numpy as np
from scipy.spatial import distance
import time

def diversify_item_feature(candidates, vectors, items, weighting = 0, numRecs = 10, weight_sigma = None):
    # weight_sigma is a 8-dimension vector, the 8 dimensions sum up to 1 
    start = time.time()
    itemID_values = candidates['item'].values
    candidates.index = pd.Index(itemID_values)
    
    if weighting != 0:
        if weight_sigma is not None :
            vectors = weight_sigma * vectors
            # weight_sigma is a 8-dimension vector, the 8 dimensions sum up to 1 
    
    vectorsDf = pd.DataFrame(vectors)
    vectorsDf.index = pd.Index(items)
    vectorsDf_in_candidate = vectorsDf[vectorsDf.index.isin(candidates.index)]
    # print(itemID_values)    
    # print(vectorsDf_in_candidate.index.to_numpy())
        # !!! row in vectorsDf_in_candidate & candidates do not correspond to the same user
    
  #==> Sorting rows of vectorsDf_in_candidate by order of candidates
    candidate_vectorsDf = vectorsDf_in_candidate.reindex(candidates.index)
    # print(candidate_vectorsDf.index.to_numpy())

    # print(candidate_vectorsDf.shape)
    
    
  #==> centroid and first candidate
    candidate_vectors = candidate_vectorsDf.to_numpy()
    items_candidate_vectors = candidate_vectorsDf.index.to_numpy()   
        # np.ndarray 
    centroid_vector = np.mean(candidate_vectors, axis = 0)
        # np.ndarray 
    
    diverse_itemIDs = []
    diverse_vectors = np.empty([0, vectors.shape[1]])
    # print(diverse_vectors.shape)
    
    firstItem_index_val = first_item(centroid_vector, candidate_vectors, items_candidate_vectors)
        # np.int64
    firstItem_vector = candidate_vectorsDf[candidate_vectorsDf.index.isin(pd.Index([firstItem_index_val]))]
        # !!! pd.Index takes an array-like (1-dimensional) variable as the first parameter
    diverse_vectors = np.concatenate((diverse_vectors, firstItem_vector.to_numpy()), axis = 0)
    diverse_itemIDs.append(firstItem_index_val)
    
    candidate_vectorsDf_left = candidate_vectorsDf.drop(pd.Index([firstItem_index_val]), axis = 0)
    # print(candidate_vectorsDf_left.shape)
    # print(diverse_vectors.shape)
    # print(diverse_itemIDs)
    # print (len(diverse_vectors))
    
  #==> Find the best next item one by one
    while len(diverse_itemIDs) < numRecs:
        nextItem_vector, nextItem_index = sum_distance(candidate_vectorsDf_left, diverse_vectors)
        candidate_vectorsDf_left = candidate_vectorsDf_left.drop(pd.Index([nextItem_index]), axis = 0)
        diverse_vectors = np.concatenate((diverse_vectors, nextItem_vector.to_numpy()), axis = 0)
            # np.ndarray 
        diverse_itemIDs.append(nextItem_index)

    # print(diverse_vectors.shape)
    # print(len(diverse_itemIDs))
    # print(diverse_itemIDs)
        # list
    # print(np.array(diverse_itemIDs))
        # make a copy
    # print(np.asarray(diverse_itemIDs))
        # work as pointer, doesn't make a copy
        
       
        
    # List -> np.ndarray
    diverse_itemIDs = np.asarray(diverse_itemIDs)
        # np.ndarray
    diverse_itemIDsDf = pd.DataFrame({'item': diverse_itemIDs})
    diverse_itemIDsDf.index = pd.Index(diverse_itemIDs)
    diverse_vectorsDf = pd.DataFrame(diverse_vectors) 
    diverse_vectorsDf.index = pd.Index(diverse_itemIDs)
    diverse_items_shuffled = candidates[candidates['item'].isin(diverse_itemIDs)]    
    recommendations = diverse_items_shuffled.reindex(diverse_itemIDsDf.index)
    #print(diverse_itemIDs)
    #print(diverse_itemIDsDf)   
    #print(recommendations)
    # print('\nSpent time in seconds: %0.2f' % (time.time() - start))
    
    return recommendations, diverse_vectorsDf
        # recommendations: ['item', 'prediction'] or ['item', 'score'] or ['item', 'discounted_score']
        # diverse_vectorsDf: : ['0', '1', ..., 'num_features']
        
def first_item(centroid, candidate_vectors, candidate_items):
    '''
    Input:
        centroid: np.ndarray
        candidate_vectors: np.ndarray
        candidate_items: np.ndarray 
    Output:
        first_index_val: the index of firstItem in np.int64
    '''
    distance_cityblock = []
    for row in candidate_vectors:
        #dist = distance.cityblock(row, centroid)
        #dist = distance.cityblock(row, centroid)**(1/2)
        #dist = distance.euclidean(row, centroid)
        #dist = distance.euclidean(row, centroid)**(1/2)
        dist = sqrt_cityblock(row, centroid)
        #dist = sqrt_cityblock(row, centroid)**(1/2)
        distance_cityblock.append(dist)
        
    # print(len(distance_cityblock))
    distance_cityblock = pd.DataFrame({'distance': distance_cityblock})
    distance_cityblock.index = pd.Index(candidate_items)
    # print(distance_cityblock.index)
    distance_cityblock_sorted = distance_cityblock.sort_values(by = 'distance', ascending = True)
    first_index_val = distance_cityblock_sorted.index[0]
            #np.int64
    return  first_index_val
        
def sum_distance(candidate_vectorsDf, diverse_set):
    '''
        candidate_vectorsDf: dataframe
        diverse_set: np.array
    '''
    #sum_dist = 0
    distance_cumulate = []
    candidate_vectors = candidate_vectorsDf.to_numpy()
    for row_candidate_vec in candidate_vectors:
        sum_dist = 0
        for row_diverse in diverse_set:
            #dist = distance.cityblock(row_candidate_vec, row_diverse)
            #dist = distance.cityblock(row_candidate_vec, row_diverse)**(1/2)
            #dist = distance.euclidean(row_candidate_vec, row_diverse)
            #dist = distance.euclidean(row_candidate_vec, row_diverse)**(1/2)
            dist = sqrt_cityblock(row_candidate_vec, row_diverse)
            #dist = sqrt_cityblock(row_candidate_vec, row_diverse)**(1/2)
            sum_dist = sum_dist + dist
        distance_cumulate.append(sum_dist)
    distance_cumulate = pd.DataFrame({'sum_distance': distance_cumulate})
    distance_cumulate.index = candidate_vectorsDf.index
    distance_cumulate_sorted = distance_cumulate.sort_values(by = 'sum_distance', ascending = False)
    bestItem_index = distance_cumulate_sorted.index[0]
        #np.int64
    bestItem_vector = candidate_vectorsDf[candidate_vectorsDf.index.isin(pd.Index([bestItem_index]))]
    
    return bestItem_vector, bestItem_index

'''    
def weight_vectors(vectors):
    # vectors: np.ndarray
    # Weight the latent feature / emotional signature vectors by Variance
        
    # axis = 0 denotes calculate the variance along each dimension of the vector
    # axis = 1 denotes calculate the variance along each vector
    sigma = np.var(vectors, axis = 0) 
        # should have same dimension with the vectors
    weighted_vectors = sigma * vectors
    
    return weighted_vectors
'''
    
def sqrt_cityblock(point1, point2):
    sqrt_city_block = 0
    for i in range(len(point1)):
        dist_axis = abs(point2[i] - point1[i])
        sqrt_city_block +=  dist_axis**(1/2)

    return sqrt_city_block    
#########################################################
########  Apply the fishingnet algorithms ###############
#########################################################
def fishingnet():
    pass