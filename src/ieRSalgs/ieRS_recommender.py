import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
         
import numpy as np
import pandas as pd
from . import setpath
import pickle
from . import diversification
from . import MF_predictor

def live_prediction(algo, liveUserID, new_ratings, item_popularity):    
    '''
        algo: trained implicitMF model
        liveUserID: str
        new_ratings: Series
        N: # of recommendations
        item_popularity: ['item', 'count', 'rank']
    '''
    items = item_popularity.item.unique()
        # items is NOT sorted
    #>>> items, rating_counts = np.unique(ratings_train['item'], return_counts = True)
        # items is sorted by default
    als_implicit_preds, liveUser_feature = algo.predict_for_user(liveUserID, items, new_ratings)
        # return a series with 'items' as the index & liveUser_feature: np.ndarray
    als_implicit_preds_df = als_implicit_preds.to_frame().reset_index()
    als_implicit_preds_df.columns = ['item', 'score']
    # print(als_implicit_preds_df.sort_values(by = 'score', ascending = False).head(10))
    
    ## discounting popular items
    highest_count = item_popularity['count'].max()
    digit = 1
    while highest_count/(10 ** digit) > 1:
        digit = digit + 1
    denominator = 10 ** digit
    # print(denominator)
    
    # a = 0.2 
    a = 0.5 # with tested data of set6
    als_implicit_preds_popularity_df = pd.merge(als_implicit_preds_df, item_popularity, how = 'left', on = 'item')
    RSSA_preds_df = als_implicit_preds_popularity_df
    RSSA_preds_df['discounted_score'] = RSSA_preds_df['score'] - a*(RSSA_preds_df['count']/denominator)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    
    # RSSA_preds_df_sorted = RSSA_preds_df.sort_values(by = 'discounted_score', ascending = False)
        
    return RSSA_preds_df, liveUser_feature

def import_trained_model(model_filename):
    f_import = open(model_filename, 'rb')
    trained_MF_model = pickle.load(f_import)
    f_import.close()
    
    return trained_MF_model